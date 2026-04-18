# Narrative RAG Pipeline - VectorDB 架構設計

本專案利用 **Weaviate (v4)** 向量資料庫與 **Gemma 4** 語言模型，打造具備「條目去重」、「分卷快照功能」以及「多重命名向量」的小說 RAG Agent 系統。

以下為目前的向量庫與儲存架構規劃指南：

## 1. 核心概念 (Core Concepts)

系統採用 **雙層向量架構**：

- **Layer 1 — `NovelChunk`**：敘事分片（scene 粒度），保存原文與摘要兩組向量。作為 RAG 的「事實來源」(ground truth)，即使 Layer 2 條目合併失敗，使用者仍可透過 chunk 向量召回原文，不會漏資訊。
- **Layer 2 — `NovelEntity`**：從 chunk 抽取出的條目（角色、道具、地點等），透過 `chunk_refs` 反指 Layer 1。條目層 merge 失敗從「災難」降級為「索引不漂亮」，不影響檢索。

兩層皆以 **Vol (卷數)** 作為切割單元，建立 **Snapshot Delta (增量快照)** 機制，確保模型在推理特定集數劇情時不會被「未來」的劇透污染。

在資料一致性上，條目的本機檔名統一依賴 **Weaviate 核發的 UUID**，解決了同一人物因別名變化（例如 Saber → 亞瑟王）導致本地 JSON 孤兒檔的痛點。Chunk 則以 `(novel_hash, vol_num, scene_index)` 生成 **deterministic UUID (uuid5)**，保證重跑 idempotent。

---

## 2. Weaviate Collection 實作結構

### 2.1 Layer 1 — `NovelChunk` (敘事分片)

| 屬性名稱       | 型別            | 說明 |
|----------------|-----------------|------|
| `novel_hash`   | `TEXT`          | 小說編號（skip_vectorization，FIELD tokenization） |
| `vol_num`      | `INT`           | 卷數 |
| `scene_index`  | `INT`           | 卷內分片序號 |
| `title`        | `TEXT`          | Scene 摘要（LLM 在 pre-process 階段生成） |
| `content`      | `TEXT`          | 原文全文 |
| `token_count`  | `INT`           | 原文 token 數 |
| `entity_refs`  | `TEXT_ARRAY`    | 此 scene 抽取到的所有 entity UUIDs（scene 結束時回填） |

Named Vectors (BYOV)：
- **`full_text`**：`content` 的向量，支援「語意搜尋原文片段」
- **`summary`**：`title` 的向量，支援「根據劇情摘要找 scene」

> ⚠️ **已知限制**：`multilingual-e5-large` 有 512-token 上限，scene 中位數 ~1844 tokens，超長部分會被截斷。目前接受此限制，後續視召回情形再決定是否換 `bge-m3` 或改為 sliding window chunking。

### 2.2 Layer 2 — `NovelEntity` (條目)

為了方便跨書系搜尋與維護單一索引，該系統將所有小說的條目都集中儲存在 `NovelEntity` 集合中，透過 Metadata 以及自帶的 CJK 分詞器保證精確查找。

### 屬性定義 (Properties)
| 屬性名稱          | 型別 (DataType)       | 說明與防呆機制 |
|-------------------|----------------------|----------------|
| `novel_hash`      | `TEXT`               | 雜湊過的小說編號，做為租戶隔離，查詢時必加 Filter。<br/>*配置 `skip_vectorization=True`，Tokenization `FIELD`* |
| `vol_num`         | `INT`                | 該屬性屬於哪一集的資訊快照設定。 |
| `entity_type`     | `TEXT`               | 例如 `character`, `item`, `poi`, `world-setting`。<br/>*Tokenization `FIELD`（整串視為單一 token，供精準 filter 使用）* |
| `keyword`         | `TEXT`               | 核心 Canonical Name。<br/>*Tokenization `FIELD`，BM25/語義比對交由 `identity` 向量處理* |
| `aliases`         | `TEXT_ARRAY` (陣列)  | 蒐羅所有代稱與方言。<br/>*Tokenization `FIELD`，供 `filter_categories`-類別的精準匹配與 `identity` 向量編碼* |
| `categories`      | `TEXT_ARRAY` (陣列)  | 標籤分類，例如 `["劍", "武器"]`。<br/>*Tokenization `FIELD`，100% 精準匹配「列出所有劍」這類列舉需求* |
| `description`     | `TEXT`               | 豐富的文本描述（含外觀、性格、功能等關鍵設定）。<br/>*Tokenization `GSE` 中文分詞器，供 hybrid BM25 模糊檢索* |
| `appeared_in`     | `INT_ARRAY` (陣列)   | 實體曾在哪些分片（Scene ID）中被提及。由程式碼維護。 |
| `chunk_refs`      | `TEXT_ARRAY` (陣列)  | 指回 Layer 1 (`NovelChunk`) 的 UUID 清單。由程式碼 union 維護。<br/>*配置 `skip_vectorization=True`，Tokenization `FIELD`* |
| `content`         | `TEXT`               | 該實體完整的極簡 JSON 字串備份。<br/>*配置 `skip_vectorization=True`* |

> 💡 **分詞器選擇原則**：
> - `description` 掛 **GSE (Go Segmenter)** 中文分詞，支援 BM25 全文模糊檢索。
> - `keyword` / `aliases` / `categories` / `entity_type` 皆用 `FIELD`（整串當一個 token），精準過濾不被切詞干擾，而其語義相似度由 `identity` 向量全權負責。

---

專案採用 **Bring Your Own Vectors (BYOV)** 設計，模型使用 `multilingual-e5-large` 進行精準編碼。Weaviate 內部維護兩組獨立的向量空間：

1. **`identity` 向量**：利用 `keyword` 與 `aliases` 計算。專門用來找人/物。
2. **`content` 向量**：利用完整的 `description` 與 `major_status_changes` 計算，專門處理模糊語義提問（如：外觀描述、性格特徵）。

### 3.2 條目去重的雙軌檢索協定 (Dual-Track Dedup Retrieval)

為了避免「名字」與「情境描述」互相污染，`search_similar_entity` 嚴格分兩條獨立軌道檢索候選人，再以「字面共字 + 相似度門檻」做程式端過濾，最後才丟給 LLM 做決策：

1. **Track A — Identity 名字軌 (純向量)**
   * 查詢文本**只用 `keyword` 本身**（不混入情境摘要，避免 BM25 被情境字眼拉偏）。
   * 走 `near_vector` 對 `identity` 向量做 cosine 比對，回傳 `distance` 並換算成 `[0,1]` 相似度。
   * 若精準類型 filter 無結果，移除 `entity_type` 過濾再試一次。

2. **Track B — Content 描述軌 (Hybrid)**
   * 查詢文本 = `keyword + query_summary`，走 `hybrid` 對 `content` 向量做 BM25+Vector 融合（`alpha=0.5`）。
   * 專門抓「別名未註冊、但描述高度重疊」的同一實體。

3. **程式端閘門 (Pre-LLM Gate)**
   以 UUID 聯集兩軌結果後，依下列優先序決定是否放行：

   | 條件 | 採用原因 |
   |------|----------|
   | `identity_sim ≥ 0.85` | 純語義已近似同名，直接放行。 |
   | `identity_sim ≥ 0.75` 且候選名稱與輸入 keyword 有字面共字 | 避免同音近似但不同物（雙重保險）。 |
   | `content_sim ≥ min_score` 且有字面共字 | 描述有吻合、名字也沾邊，值得交 LLM 判斷。 |
   | `content_sim ≥ 0.45` | 描述極度吻合，即使名字毫無關聯也放行（例如隱藏身份橋段）。 |
   | 其他 | **程式端直接剔除，不送給 LLM，防範望文生義的誤合併。** |

> 📌 **為什麼要程式端過濾？** 先前曾出現「劍」被 RAG 匹配到「儲倉」分數 0.5 的案例。根因是 Weaviate hybrid score 是 RRF 融合值，在小資料庫下排第一就近 0.5，與實際語義無關。新協定用真餘弦相似度把關，並以字面共字做硬閘門。

### 3.3 知識提取與決策分流 (Branching Decision Logic)
為了提高效率並防止邏輯混亂，系統在處理新條目時會根據 Weaviate 的檢索結果進行分流：

#### 分流 A：全新條目初始化 (Creation)
若 Weaviate 檢索不到任何相似的候選人，系統直接進入 **Initialization** 流程：
- 任務：僅負責將提取到的片段資訊豐富化，撰寫高品質的初始描述與狀態。
- 優點：LLM 不需要參與複雜的合併判斷，減少 Token 浪費與錯誤機率。

#### 分流 B：現有條目合併與更新 (Merging)
若檢索到 Top-K 候選人，則進入 **Expert Decision** 流程：
- 任務：由 LLM 判斷 `selected_index`（0-N 匹配，-1 則即使相似也視為新條目）。
- 合併原則：如果確定匹配，則執行 **Data Merge**。

#### 系統合併核心機制：
為了確保資料結構的鋼鐵級穩定，Agent 內建了自動化同步邏輯：
1. **Canonical Keyword 鎖定**：如果判定為同一實體，系統會優先保留「已存在」的名稱作為 Canonical Name，防止實體命名隨場景偏差（例如：從代號變為真名）。新名稱將自動歸入 `aliases`。
2. **別名聯集 (Alias Merging)**：自動合併新舊資料中所有的別名。
3. **重大狀態變更整合 (Status Aggregation)**：自動合併並按場景順序排列 `major_status_changes`。
4. **場景追蹤 (`appeared_in`)**：由程式碼嚴格維護，記錄實體曾在哪一卷、哪一場景出現過。
5. **Chunk 追蹤 (`chunk_refs`)**：由程式碼 union 維護，新 chunk UUID 會併入既有清單，合併兩實體時保留舊實體的 chunk_refs。

### 3.4 Per-Scene LCEL Pipeline

每個 scene 依序經過五個 Runnable step，以 LCEL 串接：

```
write_chunk_step      → Layer 1 upsert，回傳 chunk_uuid（deterministic UUID，重跑 idempotent）
    │
extract_step          → LLM 抽取 entities
    │
merge_step            → Per-entity：RAG 檢索 → LLM merge/create → inline upsert 到 Layer 2
    │                   （★ inline upsert 是修正 P1 的關鍵：同 scene 後續別名能
    │                      即時透過 RAG 看到剛寫入的前置條目）
    │
backfill_chunk_refs   → Scene 結束後，把本場所有 entity UUIDs 回填到 chunk.entity_refs
    │
save_scene_json       → 寫 `world/<type>/<uuid>.json` 與 augment `scene_XXX.json`
```

> 📌 **為什麼 inline upsert 關鍵？** 先前 upsert 延到 `_save_step` 批次執行，導致同 scene 內「迪朗達爾 / 聖劍 / 王者之劍」三個別名在 merge 期間看不見彼此（Weaviate 裡還沒寫入），結果各自被視為新條目。inline upsert 讓第二、第三個別名的 RAG 檢索能命中第一個已寫入的條目。

---

## 4. 本機磁碟的 JSON 備份協定

除了 VectorDB 的紀錄，為了後續還原、Debug 與其他 RAG 第二層系統運用，系統同樣維護一份最高高精度的本地 JSON 關聯。

#### 條目檔案 (Entity Profiles)
路徑：`output/<hash>/world/vol_<X>/<type>/<UUID>.json`
每一次 Weaviate 更新，就同步用最新的 JSON 內容將檔案複寫在本卷的資料夾下。同一卷內重複出現的人物會不斷堆疊 `appeared_in`、`chunk_refs` 與 `major_status_changes` 資訊。

#### 場景分片附加資訊 (Scene Enhancements)
路徑：`output/<hash>/scenes/processed/vol_<X>/scene_<Y>.json`
在每個分片的處理尾聲，該分片會補上自己對應的 `chunk_uuid` 與本場出現過的所有條目清單：
```json
{
  ...
  "chunk_uuid": "d1f3...-a1b2",
  "entities_extracted": [
     {
       "keyword": "阿爾托莉雅",
       "type": "character",
       "uuid": "8a92b23-....",
       "chunk_uuid": "d1f3...-a1b2"
     }
  ]
}
```
這個架構允許「Scene Agent」以 `O(1)` 精準帶著 UUID 去本機/Weaviate 抓取當時完整的條目世界觀，徹底消除因別名不同導致找不到伏筆與裝備的錯誤；同時雙向鏈結（`NovelChunk.entity_refs` ⇄ `NovelEntity.chunk_refs`）讓任何一端都可以回推另一端。

---

## 5. 診斷日誌系統 (Diagnostic Logs)

所有合併決策與檢索過程皆會產生透明化的 JSON 日誌，儲存於 `output/<novel_hash>/logs/extraction/`。

### 日誌核心欄位：
* **`rag_search_params`**：記錄當時搜尋 Weaviate 使用的關鍵字、摘要、卷數、類型與 Top-K 參數。
* **`candidates_from_rag`**：記錄 Weaviate 實際回傳的所有原始資料 (含 UUID 與 相似度評分)。
* **`thought`**：記錄 LLM 對於候選人篩選的 Chain-of-Thought 推理邏輯。
* **`result`**：記錄最終產出的合併資料結構與 `selected_index`。

透過此日誌，開發者可以輕鬆判斷「合併失效」是因為 Weaviate 沒搜到資料，還是 LLM 腦補了錯誤的匹配。

---

## 6. 檢索架構規劃 (RAG Agent 查詢藍圖)

為了避免未來實作問答 Agent（例如 LangChain Tool Calling）時造成 System Prompt 臃腫與模型選擇障礙，本專案捨棄硬寫數十種特化查表的函式，改採 **「單一高抽象萬用查詢接口 (Universal Self-Querying Tool)」** 策略。

### 單一萬用工具：`search_world_knowledge`
對話 Agent 僅需認識並呼叫這個萬用工具，依照使用者意圖靈活帶入不同參數組合，把複雜的檢索路由交給 Python 後端處理：

```json
{
  "query_text": "語意搜尋與 BM25 關鍵字用。例：'長得像精靈的女孩'。（單純列出時可留空不傳）",
  "filter_type": "可選：character, item, poi, world-setting",
  "filter_categories": "可選：用標籤陣列做精準檢索。例：['劍']",
  "sort_by": "可選：'relevance' (預設), 'appearances' (依出場數排序，找尋主角/主要設定)",
  "limit": "筆數上限 (List 列舉功能必用)"
}
```

### 多態降維切換與實戰情境
透過這種做法，這一個接口便兼具了**「條目知識圖譜搜查」**與**「傳統關聯式資料庫查詢」**的能力：

1. **混合檢索 (Hybrid Vector Search)**
   * **情境**：「男主拿過那把詛咒聖劍去過哪座城堡嗎？」
   * **運作**：LLM 填入 `query_text="男主 詛咒聖劍 城堡"`。Python 開啟 e5 的 Vector 進行語義強算抓取。
2. **精準條件過濾 (Metadata Category Filter)**
   * **情境**：「這本書出現了哪些劍？」
   * **運作**：LLM 填入 `filter_type="item", filter_categories=["劍"]`。由於 `categories` 採用 `FIELD` tokenization（整串當一個 token），可直接用 `Filter.contains_any` 做 100% 精準匹配，不會被中文分詞誤切。
3. **統計列舉功能 (List & Sort Fetching)**
   * **情境**：「請列出這本小說裡出場次數最高的主要人物前五名。」
   * **運作**：LLM 取巧判斷不需語意，傳入 `query_text="", filter_type="character", sort_by="appearances", limit=5`。Python 後端偵測到文字留空，**不走 Vector**，直接化身傳統 Database 叫出資料，並以 `appeared_in` 的涵蓋廣度作為主角排名指標，輕鬆解決大模型的計數弱點。

---

## 7. 候處理與 Context 排序策略 (RAG Post-Retrieval)

### 1. 嚴格時序排列 (Chronological/Causal Sorting)
在組裝 RAG Prompt 時，**嚴格棄用任何破壞時序的重排演算法** (如 LongContextReorder)。
無論是萃取角色生平歷史或檢索連續劇情分片，一律**依照 Vol / Scene Index 發展的先後順序進行線性排列**送入大語言模型，以此來激發模型最佳的「狀態變化與因果推理」能力。

### 2. 棄用傳統 Reranker (防範語意破壞)
絕大部份市面上的輕量級 Rerank 模型（如 `bge-reranker`）皆基於百科全書或標準問答集訓練，其設計邏輯偏向事實檢索。對於小說中極端複雜的「隱喻」、「伏筆」、「人物心境」與「文學性描述」理解能力極差。
若強悍介入這類泛用型 Reranker 進行二次打分，極容易發生**「干擾大於效果」**的慘況——把沒有直接命中關鍵字、但實際上是核心伏筆的情境分片當成雜訊剃除。
因此，只要檢索回的 Context 總量座落在 64K Token 的防守範圍內，本系統將完全信任 Weaviate Hybrid Search (`e5` 語意 + `BM25` 精準) 的初篩結果，一律無碼直通給 Gemma-4 處理，保證最深層的文學解讀不被破壞。

### 3. 超量 Context 應對：FSM 多輪篩選代理 (Map-Reduce)
若遇到諸如「總結整部小說所有大事件」這類注定突破 64K 物理上限的史詩級檢索，系統也不依賴粗暴的截斷或 Rerank。
取而代之的是啟動 **FSM (有限狀態機) Agent** 進行多段式處理：
* **Map (分塊過篩)**：將破表的 Context 依照卷數或事件鍊切分為多個安全的批次，LLM 在這幾個平行或序列狀態中，執行「是否有命中問題意圖」的獨立初篩。
* **Reduce (狀態收斂)**：將各輪過篩保留下來的高純度結點交給最終的狀態機進行整合推理。以此機制打破硬體 Context 極限，實現真正意義上的無損全本分析。
