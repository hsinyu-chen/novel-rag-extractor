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
                    Property(name="novel_hash", data_type=DataType.TEXT, skip_vectorization=True, tokenization=Tokenization.FIELD),
                    Property(name="vol_num", data_type=DataType.INT),
                    Property(name="entity_type", data_type=DataType.TEXT, tokenization=Tokenization.GSE),
                    Property(name="keyword", data_type=DataType.TEXT, tokenization=Tokenization.GSE),
                    Property(name="aliases", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.GSE),
                    Property(name="categories", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.GSE),
                    Property(name="appeared_in", data_type=DataType.INT_ARRAY),
                    Property(name="description", data_type=DataType.TEXT, tokenization=Tokenization.GSE),
                    Property(name="content", data_type=DataType.TEXT, skip_vectorization=True) 
                ],
                # 宣告多組獨立向量空間 (BYOV 模式)
                vectorizer_config=[
                    Configure.NamedVectors.none(name="identity"),
                    Configure.NamedVectors.none(name="content")
                ],
                # 徹底關閉 GSE 的停用詞過濾，允許所有切詞進行 BM25 比對
                inverted_index_config=Configure.inverted_index(
                    stopwords_preset=StopwordsPreset.NONE
                )
            )

    def _generate_entity_vectors(self, data_dict: dict) -> Dict[str, List[float]]:
        """
        根據極簡統一結構生成 Named Vectors。
        """
        keyword = data_dict.get("keyword", "")
        aliases = data_dict.get("aliases", [])
        aliases_str = ", ".join(aliases or [])
        identity_text = f"名稱：{keyword}，別名：{aliases_str}"
        
        description = data_dict.get("description", "")
        # 加入重大狀態變更作為語義補充
        status_list = data_dict.get("major_status_changes", [])
        status_text = "。狀態更新：" + "；".join([s.get("event", "") for s in status_list]) if status_list else ""
        content_text = f"{description}{status_text}"

        # 這裡的 embed_documents 會自動在前面幫我們加上 'passage: '
        vecs = self.embed_func.embed_documents([identity_text, content_text])
        return {
            "identity": vecs[0],
            "content": vecs[1]
        }

    def search_similar_entity(self, novel_hash: str, max_vol: int, entity_type: str, keyword: str, query_summary: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        利用 Hybrid Search (Text BM25 + Vector) 到 Weaviate 進行查表，找出候選名單。
        具備二段式檢索：首波嘗試準確類型過濾，失敗則放寬限制。
        """
        collection = self._client.collections.get("NovelEntity")
        
        # 強化 Search Text 加入部分背景，避免純 Keyword 觸發 Stopword 錯誤
        search_text = f"{keyword} {query_summary[:200]}".strip()
        if not search_text:
             return []

        # 自動處理 'query: ' 前綴
        query_vector = self.embed_func.embed_query(search_text)
        
        # 基礎過濾條件：小說 Hash 與 卷數限制
        base_filter = Filter.by_property("novel_hash").equal(novel_hash) & \
                      Filter.by_property("vol_num").less_or_equal(max_vol)
        
        # 第一階段：嘗試精準類型過濾 (Strict Pass)
        strict_filter = base_filter
        if entity_type:
             strict_filter = strict_filter & Filter.by_property("entity_type").equal(entity_type)

        try:
            # 第一波嘗試：Alpha=0.5 (平衡模式)
            response = collection.query.hybrid(
                query=search_text,
                vector=query_vector,
                target_vector="content",
                alpha=0.5,
                filters=strict_filter,
                limit=top_k,
                return_metadata=MetadataQuery(score=True)
            )

            # 第二階段：若精準過濾沒結果，執行回退檢索 (Lenient Pass)
            if not response.objects:
                # 放寬搜尋：Alpha=0.3 (偏重關鍵字)，且不限 entity_type
                response = collection.query.hybrid(
                    query=search_text,
                    vector=query_vector,
                    target_vector="context",
                    alpha=0.3,
                    filters=base_filter,
                    limit=top_k,
                    return_metadata=MetadataQuery(score=True)
                )

            # 整理回傳格式
            results = []
            for obj in response.objects:
                content_json = obj.properties.get("content", "{}")
                data = json.loads(content_json)
                data["_weaviate_uuid"] = str(obj.uuid)
                data["_score"] = obj.metadata.score
                results.append(data)
            
            return results

        except Exception as e:
            # 最後保險：純向量檢索 (應對 Stopwords 報錯)
            print(f"[Weaviate] Hybrid search failed ({e}), falling back to vector search...")
            try:
                response = collection.query.near_vector(
                    near_vector=query_vector,
                    target_vector="context",
                    filters=base_filter,
                    limit=top_k,
                    return_metadata=MetadataQuery(score=True)
                )
                results = []
                for obj in response.objects:
                    content_json = obj.properties.get("content", "{}")
                    data = json.loads(content_json)
                    data["_weaviate_uuid"] = str(obj.uuid)
                    data["_score"] = obj.metadata.score
                    results.append(data)
                return results
            except Exception as e2:
                print(f"[Weaviate] Critical error in search: {e2}")
                return []

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
            "description": data_dict.get("description", ""),
            "aliases": data_dict.get("aliases", []),
            "categories": data_dict.get("categories", []),
            "appeared_in": appeared_in,
            "content": json.dumps(data_dict, ensure_ascii=False)
        }

        # 如果有 existing_uuid，我們使用 replace 確保不產生重複
        if existing_uuid:
            collection.data.replace(
                uuid=existing_uuid,
                properties=properties,
                vector=named_vectors
            )
            return existing_uuid
        else:
            # 建立全新 UUID 並 insert
            target_uuid = str(uuid.uuid4())
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
