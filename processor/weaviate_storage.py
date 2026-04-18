import uuid
import json
from typing import List, Dict, Optional, Any
import weaviate
from weaviate.classes.config import Property, DataType, Configure, Tokenization, StopwordsPreset
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.data import DataObject

# 固定 namespace，用於從 (novel_hash, vol_num, scene_index) 生成 deterministic chunk UUID
_CHUNK_UUID_NAMESPACE = uuid.UUID("6f3a1e4c-2a93-4b5e-9c3a-1c4d9e7f0a11")


class WeaviateStorage:
    """
    Weaviate 向量庫封裝
    雙層架構：
      - Layer 1 (NovelChunk)：敘事分片，存原文與摘要向量
      - Layer 2 (NovelEntity)：抽取出的條目，透過 chunk_refs 指回 Layer 1
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
        self._ensure_collections()

    def _ensure_collections(self):
        self._ensure_entity_collection()
        self._ensure_chunk_collection()

    def _ensure_entity_collection(self):
        collection_name = "NovelEntity"
        if not self._client.collections.exists(collection_name):
            print(f"Creating Weaviate Collection '{collection_name}' with Named Vectors...")
            self._client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="novel_hash", data_type=DataType.TEXT, skip_vectorization=True, tokenization=Tokenization.FIELD),
                    Property(name="vol_num", data_type=DataType.INT),
                    Property(name="entity_type", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="keyword", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="aliases", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD),
                    Property(name="categories", data_type=DataType.TEXT_ARRAY, tokenization=Tokenization.FIELD),
                    Property(name="appeared_in", data_type=DataType.INT_ARRAY),
                    Property(name="chunk_refs", data_type=DataType.TEXT_ARRAY, skip_vectorization=True, tokenization=Tokenization.FIELD),
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

    def _ensure_chunk_collection(self):
        collection_name = "NovelChunk"
        if not self._client.collections.exists(collection_name):
            print(f"Creating Weaviate Collection '{collection_name}' with Named Vectors...")
            self._client.collections.create(
                name=collection_name,
                properties=[
                    Property(name="novel_hash", data_type=DataType.TEXT, skip_vectorization=True, tokenization=Tokenization.FIELD),
                    Property(name="vol_num", data_type=DataType.INT),
                    Property(name="scene_index", data_type=DataType.INT),
                    Property(name="title", data_type=DataType.TEXT, tokenization=Tokenization.GSE),
                    Property(name="content", data_type=DataType.TEXT, tokenization=Tokenization.GSE),
                    Property(name="token_count", data_type=DataType.INT),
                    Property(name="entity_refs", data_type=DataType.TEXT_ARRAY, skip_vectorization=True, tokenization=Tokenization.FIELD),
                ],
                vectorizer_config=[
                    Configure.NamedVectors.none(name="full_text"),
                    Configure.NamedVectors.none(name="summary"),
                ],
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

    def search_similar_entity(self, novel_hash: str, max_vol: int, entity_type: str, keyword: str, query_summary: str, top_k: int = 5, min_score: float = None, identity_strong: float = None, identity_keep: float = None, content_strong: float = None) -> List[Dict[str, Any]]:
        """
        雙軌檢索：
          - Track A (identity)：純 keyword 走 near_vector，走 cosine 找同名 / 同人。
          - Track B (content)：以情境摘要走 hybrid (BM25+vector)，抓別名未註冊但描述重疊的同一實體。
        兩軌結果聯集後，透過「字面共字 + 相似度門檻」做程式端過濾，避免語義無關的條目被送進 LLM。

        門檻參數（若未傳入則從 config 讀取）：
          identity_strong: identity 純語義強配（直接放行）
          identity_keep:   identity 中度語義 + 字面共字 才放行
          content_strong:  content 描述強配（即使無字面共字也放行）
          min_score:       content + 字面共字 的最低門檻
        """
        if not keyword:
            return []

        collection = self._client.collections.get("NovelEntity")

        base_filter = Filter.by_property("novel_hash").equal(novel_hash) & \
                      Filter.by_property("vol_num").less_or_equal(max_vol)
        strict_filter = base_filter
        if entity_type:
            strict_filter = strict_filter & Filter.by_property("entity_type").equal(entity_type)

        bucket: Dict[str, Dict[str, Any]] = {}

        # Track A：identity near_vector，查詢只用 keyword 本身，不混入情境摘要
        try:
            kw_vector = self.embed_func.embed_query(keyword)
            resp_a = collection.query.near_vector(
                near_vector=kw_vector,
                target_vector="identity",
                filters=strict_filter,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True)
            )
            for obj in resp_a.objects:
                dist = obj.metadata.distance if obj.metadata.distance is not None else 2.0
                sim = max(0.0, 1.0 - dist / 2.0)  # cosine distance → [0,1] certainty
                self._accumulate_candidate(bucket, obj, identity_sim=sim)

            # 放寬：若精準類型過濾沒命中，移除 type filter 再試一次 identity
            if not bucket and entity_type:
                resp_a2 = collection.query.near_vector(
                    near_vector=kw_vector,
                    target_vector="identity",
                    filters=base_filter,
                    limit=top_k,
                    return_metadata=MetadataQuery(distance=True)
                )
                for obj in resp_a2.objects:
                    dist = obj.metadata.distance if obj.metadata.distance is not None else 2.0
                    sim = max(0.0, 1.0 - dist / 2.0)
                    self._accumulate_candidate(bucket, obj, identity_sim=sim)
        except Exception as e:
            print(f"[Weaviate] Track A (identity) failed: {e}")

        # Track B：content hybrid，查詢用情境摘要 + keyword 供 BM25 命中
        summary = (query_summary or "").strip()[:400]
        if summary:
            try:
                content_query = f"{keyword} {summary}".strip()
                content_vector = self.embed_func.embed_query(content_query)
                resp_b = collection.query.hybrid(
                    query=content_query,
                    vector=content_vector,
                    target_vector="content",
                    alpha=0.5,
                    filters=strict_filter,
                    limit=top_k,
                    return_metadata=MetadataQuery(score=True)
                )
                for obj in resp_b.objects:
                    score = obj.metadata.score if obj.metadata.score is not None else 0.0
                    self._accumulate_candidate(bucket, obj, content_sim=score)
            except Exception as e:
                print(f"[Weaviate] Track B (content) failed: {e}")

        # 字面共字 + 相似度門檻過濾
        IDENTITY_STRONG = identity_strong if identity_strong is not None else float(self.conf.get("rag_identity_strong", 0.75))
        IDENTITY_KEEP   = identity_keep   if identity_keep   is not None else float(self.conf.get("rag_identity_keep",   0.62))
        CONTENT_STRONG  = content_strong  if content_strong  is not None else float(self.conf.get("rag_content_strong",  0.35))
        MIN_SCORE       = min_score       if min_score       is not None else float(self.conf.get("rag_content_min",     0.10))
        kw_chars = set(keyword)

        results: List[Dict[str, Any]] = []
        for uuid_key, entry in bucket.items():
            data = entry["data"]
            id_sim = entry.get("identity_sim", 0.0)
            content_sim = entry.get("content_sim", 0.0)

            cand_names = [data.get("keyword", "")] + list(data.get("aliases") or [])
            name_chars = set("".join(n for n in cand_names if isinstance(n, str)))
            has_overlap = bool(kw_chars & name_chars)

            match_track = None
            score = 0.0
            if id_sim >= IDENTITY_STRONG:
                match_track, score = "identity-strong", id_sim
            elif has_overlap and id_sim >= IDENTITY_KEEP:
                match_track, score = "identity+literal", id_sim
            elif has_overlap and content_sim >= MIN_SCORE:
                match_track, score = "content+literal", content_sim
            elif content_sim >= CONTENT_STRONG:
                match_track, score = "content-strong", content_sim

            if match_track is None:
                continue

            data["_score"] = score
            data["_identity_sim"] = id_sim
            data["_content_sim"] = content_sim
            data["_match_track"] = match_track
            results.append(data)

        results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return results[:top_k]

    def _accumulate_candidate(self, bucket: Dict[str, Dict[str, Any]], obj, identity_sim: float = None, content_sim: float = None):
        """
        以 uuid 為 key 合併兩軌檢索結果，同軌取較高分。
        """
        uuid_key = str(obj.uuid)
        entry = bucket.get(uuid_key)
        if entry is None:
            content_json = obj.properties.get("content", "{}")
            try:
                data = json.loads(content_json)
            except Exception:
                data = {}
            data["_weaviate_uuid"] = uuid_key
            entry = {"data": data, "identity_sim": 0.0, "content_sim": 0.0}
            bucket[uuid_key] = entry
        if identity_sim is not None:
            entry["identity_sim"] = max(entry["identity_sim"], identity_sim)
        if content_sim is not None:
            entry["content_sim"] = max(entry["content_sim"], content_sim)

    def upsert_entity(self, novel_hash: str, vol_num: int, data_dict: dict, existing_uuid: str = None, scene_idx: int = 0, chunk_uuid: str = None) -> str:
        """
        將最新的合併檔案，利用 BYOV 寫入 Weaviate
        以 `vol_num` 為單位不覆蓋舊卷的，如果有 existing_uuid，是否該保留同一個 UUID？
        在 Snapshot 架構中，每個 volume 應該要有自己的一筆 object，才能達成分開保存。

        chunk_uuid (optional)：本次 scene 對應的 NovelChunk UUID，會被 union 進 chunk_refs。
        """
        collection = self._client.collections.get("NovelEntity")

        # 計算多重向量
        named_vectors = self._generate_entity_vectors(data_dict)

        # 維護 appeared_in 陣列
        appeared_in = data_dict.get("appeared_in", [])
        if scene_idx > 0 and scene_idx not in appeared_in:
            appeared_in.append(scene_idx)
            data_dict["appeared_in"] = appeared_in

        # 維護 chunk_refs：與現有 list 做 union，保持順序
        chunk_refs = list(data_dict.get("chunk_refs", []) or [])
        if chunk_uuid and chunk_uuid not in chunk_refs:
            chunk_refs.append(chunk_uuid)
        data_dict["chunk_refs"] = chunk_refs

        properties = {
            "novel_hash": novel_hash,
            "vol_num": vol_num,
            "entity_type": data_dict.get("type", "unknown"),
            "keyword": data_dict.get("keyword", ""),
            "description": data_dict.get("description", ""),
            "aliases": data_dict.get("aliases", []),
            "categories": data_dict.get("categories", []),
            "appeared_in": appeared_in,
            "chunk_refs": chunk_refs,
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

    # ===== Layer 1 (NovelChunk) =====

    def chunk_uuid(self, novel_hash: str, vol_num: int, scene_idx: int) -> str:
        """
        從 (novel_hash, vol_num, scene_index) 生成 deterministic UUID，保證重跑 idempotent。
        """
        key = f"{novel_hash}:vol{vol_num}:scene{scene_idx}"
        return str(uuid.uuid5(_CHUNK_UUID_NAMESPACE, key))

    def _generate_chunk_vectors(self, title: str, content: str) -> Dict[str, List[float]]:
        """
        Layer 1 用雙向量：
          full_text：整段原文（受 e5 512-token 上限影響，暫不處理）
          summary：scene 既有的 title 欄位（本身就是 LLM 生成的摘要）
        """
        full_text = content or ""
        summary_text = title or ""
        vecs = self.embed_func.embed_documents([full_text, summary_text])
        return {
            "full_text": vecs[0],
            "summary": vecs[1],
        }

    def upsert_chunk(self, novel_hash: str, vol_num: int, scene_idx: int, title: str, content: str, token_count: int = 0) -> str:
        """
        寫入 / 覆寫單一 scene chunk。UUID 由 (novel_hash, vol, scene_idx) 決定，重跑 idempotent。
        回傳 chunk UUID。entity_refs 不在此處寫入，由 _merge_step 完成後呼叫 update_chunk_entity_refs 回填。
        """
        collection = self._client.collections.get("NovelChunk")
        target_uuid = self.chunk_uuid(novel_hash, vol_num, scene_idx)
        named_vectors = self._generate_chunk_vectors(title, content)
        properties = {
            "novel_hash": novel_hash,
            "vol_num": vol_num,
            "scene_index": scene_idx,
            "title": title or "",
            "content": content or "",
            "token_count": int(token_count or 0),
            "entity_refs": [],
        }
        if collection.data.exists(target_uuid):
            collection.data.replace(uuid=target_uuid, properties=properties, vector=named_vectors)
        else:
            collection.data.insert(uuid=target_uuid, properties=properties, vector=named_vectors)
        return target_uuid

    def update_chunk_entity_refs(self, chunk_uuid: str, entity_uuids: List[str]):
        """
        Scene 所有 entity 寫入完成後，回填 entity UUID 清單到對應 chunk。
        只更新 entity_refs 欄位，不動向量。
        """
        if not chunk_uuid:
            return
        collection = self._client.collections.get("NovelChunk")
        dedup = []
        seen = set()
        for u in entity_uuids or []:
            if u and u not in seen:
                seen.add(u)
                dedup.append(u)
        try:
            collection.data.update(
                uuid=chunk_uuid,
                properties={"entity_refs": dedup}
            )
        except Exception as e:
            print(f"[Weaviate] Failed to update chunk entity_refs for {chunk_uuid}: {e}")

    def clear_novel_volume(self, novel_hash: str, vol_num: int):
        """
        清除 Weaviate 中特定小說某卷的所有條目 (NovelEntity) 與分片 (NovelChunk)。
        """
        entity_col = self._client.collections.get("NovelEntity")
        chunk_col = self._client.collections.get("NovelChunk")
        vol_filter = Filter.by_property("novel_hash").equal(novel_hash) & Filter.by_property("vol_num").equal(vol_num)
        try:
            entity_col.data.delete_many(where=vol_filter)
            print(f"  [Weaviate] Cleared NovelEntity for Hash={novel_hash}, Vol={vol_num}")
        except Exception as e:
            print(f"  [Weaviate] Error clearing entity volume: {e}")
        try:
            chunk_col.data.delete_many(where=vol_filter)
            print(f"  [Weaviate] Cleared NovelChunk for Hash={novel_hash}, Vol={vol_num}")
        except Exception as e:
            print(f"  [Weaviate] Error clearing chunk volume: {e}")

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
