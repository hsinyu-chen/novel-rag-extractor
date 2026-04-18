import uuid
import json
from typing import List, Dict, Optional, Any
import weaviate
from weaviate.classes.config import Property, DataType, Configure, Tokenization, StopwordsPreset
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.data import DataObject

class WeaviateStorage:
    """
    Weaviate 向量庫封裝
    負責連線管理與 NovelEntity Collection 面向存取 (包含 Named Vectors 操作)
    """
    def __init__(self, config, embed_func):
        self.conf = config
        self.embed_func = embed_func # A callable or object with embed_query/embed_documents
        self._client = None
        self._init_client()

    def _init_client(self):
        host = self.conf.get("weaviate_host", "localhost")
        http_port = self.conf.get("weaviate_http_port", 8080)
        http_secure = self.conf.get("weaviate_http_secure", False)
        grpc_port = self.conf.get("weaviate_grpc_port", 50051)
        grpc_secure = self.conf.get("weaviate_grpc_secure", False)
        
        print(f"Connecting to Weaviate at {host}...")
        self._client = weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=host,
            grpc_port=grpc_port,
            grpc_secure=grpc_secure
        )
        self._ensure_collection()

    def _ensure_collection(self):
        collection_name = "NovelEntity"
        if not self._client.collections.exists(collection_name):
            print(f"Creating Weaviate Collection '{collection_name}' with Named Vectors...")
            self._client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="novel_hash", data_type=DataType.TEXT, skip_vectorization=True),
                    Property(name="vol_num", data_type=DataType.INT),
                    Property(name="entity_type", data_type=DataType.TEXT, tokenization=Tokenization.GSE),
                    Property(name="keyword", data_type=DataType.TEXT, tokenization=Tokenization.GSE),
                    Property(name="aliases", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.GSE),
                    Property(name="categories", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.GSE),
                    Property(name="appeared_in", data_type=DataType.INT_ARRAY),
                    Property(name="content", data_type=DataType.TEXT, skip_vectorization=True) # Full JSON dumped
                ],
                # 宣告多組獨立向量空間 (BYOV 模式)
                vectorizer_config=[
                    Configure.NamedVectors.none(name="identity"),
                    Configure.NamedVectors.none(name="equipment"),
                    Configure.NamedVectors.none(name="context")
                ],
                # 路線一：徹底關閉 GSE 的停用詞過濾，允許所有切詞進行 BM25 比對
                inverted_index_config=Configure.inverted_index(
                    stopwords_preset=StopwordsPreset.NONE
                )
            )

    def _generate_entity_vectors(self, data_dict: dict) -> Dict[str, List[float]]:
        """
        根據輸入的整理好的 Dictionary，調用 Embed Engine 生成 Named Vectors。
        注意: multilingual-e5 需要在端點加上 passage:
        """
        keyword = data_dict.get("keyword", "")
        aliases = data_dict.get("aliases", [])
        aliases_str = ", ".join(aliases)
        identity_text = f"名稱：{keyword}，別名：{aliases_str}"
        
        # 設備描述：如果是道具本身就是道具，如果是腳色抓 profile.equipment
        equipment_text = ""
        if data_dict.get("type") == "character" and "profile" in data_dict:
            equip_list = data_dict["profile"].get("equipment", "")
            if isinstance(equip_list, list):
                equipment_text = ", ".join(equip_list)
            else:
                equipment_text = str(equip_list)
        elif data_dict.get("type") == "item":
            equipment_text = data_dict.get("description", keyword)

        if not equipment_text.strip():
            equipment_text = "無裝備"

        # 背景長文描述
        context_text = ""
        if data_dict.get("type") == "character" and "profile" in data_dict:
            p = data_dict["profile"]
            context_text = f"行動準則: {p.get('action_principles','')}。個性: {p.get('personality','')}。外觀: {p.get('appearance','')}"
        else:
            context_text = data_dict.get("description", identity_text)

        # 這裡的 embed_documents 會自動在前面幫我們加上 'passage: '
        vecs = self.embed_func.embed_documents([identity_text, equipment_text, context_text])
        return {
            "identity": vecs[0],
            "equipment": vecs[1],
            "context": vecs[2]
        }

    def search_similar_entity(self, novel_hash: str, max_vol: int, entity_type: str, keyword: str, query_summary: str) -> Optional[Dict[str, Any]]:
        """
        利用 Hybrid Search (Text BM25 + Vector) 到 Weaviate 進行查表，找出最相似條目的舊紀錄。
        Weaviate v4 語法。
        """
        collection = self._client.collections.get("NovelEntity")
        
        search_text = f"{keyword} {query_summary}".strip()
        if not search_text:
             return None

        # Query Engine 需要 'query: ' 作為前綴，我們自建的 embed_query 已經有自動加
        query_vector = self.embed_func.embed_query(search_text)
        
        # Weaviate 支援 Query 多個 target_vectors。這裡我們先挑 context vector。
        # 如果是明確在找道具也可以查 equipment。為了廣泛匹配，我們使用 context。
        
        filter_obj = Filter.by_property("novel_hash").equal(novel_hash) & \
                     Filter.by_property("vol_num").less_or_equal(max_vol)
        
        # 我們如果知道對方抽出來的條目 Type 是啥，可以限定搜尋範圍，減少跨界誤差
        if entity_type:
             filter_obj = filter_obj & Filter.by_property("entity_type").equal(entity_type)

        try:
            response = collection.query.hybrid(
                query=search_text,
                vector=query_vector,
                target_vector="context", # 指定查詢在 context 這個 Named Vector 空間
                alpha=0.5, # 50% BM25, 50% Vector
                filters=filter_obj,
                limit=1,
                return_metadata=MetadataQuery(score=True)
            )

            if not response.objects:
                return None
                
            best_match = response.objects[0]
            # 解除 JSON 回傳
            content_json = best_match.properties.get("content", "{}")
            
            data = json.loads(content_json)
            # 把 Weaviate 原生分配的 UUID 也帶出去，方便 update
            data["_weaviate_uuid"] = str(best_match.uuid)
            return data
        except Exception:
            # 如果因為全部都是 Stopwords 導致 BM25 報錯，我們直接降級為純粹的 Vector Search
            try:
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    target_vector="context",
                    filters=filter_obj,
                    limit=1,
                    return_metadata=MetadataQuery(score=True)
                )
                
                if not response.objects:
                    return None
                    
                best_match = response.objects[0]
                content_json = best_match.properties.get("content", "{}")
                data = json.loads(content_json)
                data["_weaviate_uuid"] = str(best_match.uuid)
                return data
            except Exception:
                return None

    def upsert_entity(self, novel_hash: str, vol_num: int, data_dict: dict, existing_uuid: str = None, scene_idx: int = 0) -> str:
        """
        將最新的合併檔案，利用 BYOV 寫入 Weaviate
        以 `vol_num` 為單位不覆蓋舊卷的，如果有 existing_uuid，是否該保留同一個 UUID？
        在 Snapshot 架構中，每個 volume 應該要有自己的一筆 object，才能達成分開保存。
        """
        collection = self._client.collections.get("NovelEntity")
        
        # 計算多重向量
        named_vectors = self._generate_entity_vectors(data_dict)
        
        # 維護 appeared_in 陣列
        appeared_in = data_dict.get("appeared_in", [])
        if scene_idx > 0 and scene_idx not in appeared_in:
            appeared_in.append(scene_idx)
            data_dict["appeared_in"] = appeared_in
            
        properties = {
            "novel_hash": novel_hash,
            "vol_num": vol_num,
            "entity_type": data_dict.get("type", "unknown"),
            "keyword": data_dict.get("keyword", ""),
            "aliases": data_dict.get("aliases", []),
            "categories": data_dict.get("categories", []),
            "appeared_in": appeared_in,
            "content": json.dumps(data_dict, ensure_ascii=False)
        }

        # 如果要覆蓋同卷的（同一個人物在同一卷不斷被更新），我們必須尋找此卷存不存在，用指定 UUID
        # 如果傳入了 existing_uuid，且是同一集的，我們就直接取代。
        target_uuid = existing_uuid if existing_uuid else str(uuid.uuid4())
        
        # 使用 data.insert 或 replace
        # Weaviate collections.data.insert 會新增，如果 uuid 重複的話如果用 replace 會覆蓋
        collection.data.insert(
            uuid=target_uuid,
            properties=properties,
            vector=named_vectors
        )
        return target_uuid

    def clear_novel_volume(self, novel_hash: str, vol_num: int):
        """
        清除 Weaviate 中特定小說某卷的所有條目資料
        """
        collection = self._client.collections.get("NovelEntity")
        try:
            collection.data.delete_many(
                where=Filter.by_property("novel_hash").equal(novel_hash) & Filter.by_property("vol_num").equal(vol_num)
            )
            print(f"  [Weaviate] Cleared NovelEntity for Hash={novel_hash}, Vol={vol_num}")
        except Exception as e:
            print(f"  [Weaviate] Error clearing volume: {e}")

    def get_existing_entity_types(self) -> List[str]:
        """
        取得資料庫目前已經收錄過的所有 entity_type 列表 (減少 LLM 重複造詞)
        """
        collection = self._client.collections.get("NovelEntity")
        try:
            res = collection.aggregate.over_all(group_by="entity_type")
            types = [g.grouped_by.value for g in res.groups]
            return [t for t in types if t and isinstance(t, str)]
        except Exception as e:
            return ["character", "item", "poi", "world-setting"]

    def close(self):
        if self._client:
            self._client.close()
