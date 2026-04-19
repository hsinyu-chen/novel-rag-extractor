# Narrative RAG Pipeline - VectorDB 架構設計

本專案利用 **Weaviate (v4)** 向量資料庫與 **Gemma 4** 語言模型，打造具備「條目去重」、「分卷快照功能」以及「多重命名向量」的小說 RAG Agent 系統。

硬體目標為8GB VRAM的設備

---

## 展示 (Demo)

以下為 QA Agent 針對已建立索引的輕小說進行跨庫跨卷問答的實際執行片段。

### 範例 1：角色身份與能力查詢

```
> uv run python main.py --mode qa --prompt "'爆肝'的主角是誰? 都有哪些強大的能力?"

>> QA Agent  scope=全庫跨 DB
─────────────────────────────── Question ───────────────────────────────
'爆肝'的主角是誰? 都有哪些強大的能力?
─────────────────────────────────────────────────────────────────────────
agent 正在思考...
agent 正在查詢 search_world_knowledge...
agent 正在查詢 search_world_knowledge...
agent 正在查詢 search_world_knowledge...
agent 正在查詢 search_scenes...
agent 正在整理回答...
──────────────────────────────── Answer ────────────────────────────────
╭───────────────────────────────────────────────────────────────────────╮
│ 主角身份                                                              │
│                                                                       │
│  • 姓名與身份：主角是佐藤(Sato)，一位被描述為「工蟻」的日本人。       │
│    他原本是一位在現實世界中擔任「爆肝工程師」的角色，後來因某些       │
│    原因被捲入了異世界奇幻設定中。                                     │
│                                                                       │
│ 強大能力與地位                                                        │
│                                                                       │
│ 根據資料庫的描述,主角具備以下能力和獲得的稱號:                        │
│                                                                       │
│  • 稱號:在克服重大危機後,主角獲得了多個重要稱號,包括「勇者」、        │
│    「迷宮破除者」和「與惡魔共舞者」。                                 │
│  • 系統介面:主角具有AR介面(Augmented Reality interface)的特性,        │
│    並且能看到時間顯示在常態主選單上。                                 │
│  • 戰鬥能力:                                                          │
│     • 能夠射出箭矢(射出第一箭)。                                      │
│     • 能夠使用魔法槍,成功對強大的上級魔族造成了命中。                 │
╰───────────────────────────────────────────────────────────────────────╯
```

### 範例 2：裝備列舉查詢

```
> uv run python main.py --mode qa --prompt "'爆肝'的主角都有些什麼裝備?"

>> QA Agent  scope=全庫跨 DB
─────────────────────────────── Question ───────────────────────────────
'爆肝'的主角都有些什麼裝備?
─────────────────────────────────────────────────────────────────────────
agent 正在思考...
agent 正在查詢 search_world_knowledge...
agent 正在查詢 search_world_knowledge...
agent 正在查詢 search_world_knowledge...
agent 正在整理回答...
──────────────────────────────── Answer ────────────────────────────────
╭───────────────────────────────────────────────────────────────────────╮
│ 根據資料庫查詢結果,關於主角所使用的裝備有以下幾項:                    │
│                                                                       │
│ 武器:                                                                 │
│                                                                       │
│  • 劍: 這是一把主角從腰間劍鞘中拔出的武器。它適用於近戰格鬥,          │
│    同時也具備極高的投擲能力,能被主角以異常速度投擲出去。              │
│  • 聖槍(光之槍/神聖之槍): 這是一種神聖的光之武器,彈藥以魔力為基礎。   │
│    它曾被主角設定至最大威力進行狙擊,用於對抗強大上級魔族。            │
│  • 聖劍(王者之劍/光之劍): 這是一種由神明授予的護國聖劍,能與魔王戰鬥。 │
│    它具有極高的戰鬥價值和特殊意義,曾被用於對抗上級魔族。              │
╰───────────────────────────────────────────────────────────────────────╯
```

### 範例 3：劇情挫折與作品調性判讀

```
> uv run python main.py --mode qa --prompt "'爆肝'的主角在第一集中遇到什麼挫折?這是純爽文嗎"

>> QA Agent  scope=全庫跨 DB
─────────────────────────────── Question ───────────────────────────────
'爆肝'的主角在第一集中遇到什麼挫折?這是純爽文嗎
─────────────────────────────────────────────────────────────────────────
agent 正在思考...
agent 正在查詢 search_scenes...
agent 正在查詢 search_world_knowledge...
agent 正在整理回答...
──────────────────────────────── Answer ────────────────────────────────
╭───────────────────────────────────────────────────────────────────────╮
│ 主角在第一章的挫折                                                    │
│                                                                       │
│ 主角在第一集中引發大規模隕石暴降以攻擊敵人後,遭遇了劇痛,              │
│ 隨後失去意識,並被隨之而來的沙塵巨浪吞沒。                             │
│                                                                       │
│ 關於是否為純爽文                                                      │
│                                                                       │
│ 根據資料庫內容判斷,這部作品不屬於純爽文。在第一卷中,                  │
│ 主角的經歷包含以下非純爽的元素:                                       │
│                                                                       │
│  • 負面事件與傷害: 主角在發動隕石攻擊後,遭受了劇痛並失去了意識。      │
│  • 殘暴的衝突: 在城市廣場上,有魔族伍斯撕裂了胖子神官長,               │
│    導致主角因眼前的殘暴行為而喪失了反應。                             │
│  • 社會層面的威脅: 主角遭遇了神官利用「積功德」等名義                 │
│    煽動民眾攻擊獸娘的暴動事件,顯示了社會和政治層面的陰謀與問題。      │
│  • 持續的致命威脅: 即使魔王已被討伐,當地人也得知                      │
│    真正的威脅是可怕的飛龍。                                           │
╰───────────────────────────────────────────────────────────────────────╯
```

上述範例實際體現了本系統的幾個核心設計價值：

- **Agent 自主多輪檢索**：QA Agent 自行決定連續呼叫 `search_world_knowledge` 多次 + `search_scenes` 一次，將條目檢索與原文 scene 檢索交叉比對後再整合回答。
- **跨書系全庫問答**：`scope=全庫跨 DB` 表示主角雖不知道「爆肝」指的是哪一本書，系統仍能透過 `novel_hash` 過濾自動鎖定該作品 (《爆肝工程師的異世界狂想曲》) 並回溯對應條目。
- **條目層 + Scene 層雙軌召回**：稱號、介面、戰鬥能力分別散落於不同 Entity 與原文分片中，靠 Layer 1/Layer 2 的雙向鏈結 (`chunk_refs` ⇄ `entity_refs`) 才能拼出完整能力輪廓。

> 系統還在調教中，目前查詢效果還可以再優化。

---

## 配置與使用說明

> **本專案尚未完全完成，部分功能（特別是條目合併與 QA Agent 多輪推理）在邊緣題材下可能出現不穩定行為，請以實驗性工具視之。**

### 前置需求

- Python **3.11+**，建議使用 [`uv`](https://docs.astral.sh/uv/) 管理依賴。
- 一組 **Weaviate v4** server , 必須啟用GSE（本機 Docker 或遠端皆可）。
- **OpenAI 相容 API** 端點各一，分別供 LLM (Gemma 4 等 chat model) 與 Embedding (`multilingual-e5-large`) 使用。本機可以 [`llama-server`](https://github.com/ggml-org/llama.cpp) 起雙 port。
- 硬體建議：**8GB VRAM 以上** GPU（用於 Gemma-4 E4B Q4_K_M + e5-large f16 同機跑）。

### 安裝

```bash
git clone <repo>
cd <repo path>
uv sync
```

### 環境變數 (`.env`)

於專案根目錄建立 `.env`，可視需求覆寫以下變數（全部為選填，表列為預設值）：

| 類別 | 變數 | 預設值 | 說明 |
|------|------|--------|------|
| Embedding | `EMBED_BASE_URL` | `http://127.0.0.1:8081/v1` | Embedding API 端點 |
|  | `EMBED_API_KEY` | `no-key-required` | API Key |
|  | `EMBED_MODEL` | `multilingual-e5-large-f16` | 模型名 |
| LLM | `SUMMARY_BASE_URL` | `http://127.0.0.1:8080/v1` | Chat LLM 端點（摘要 / 抽取 / 合併共用） |
|  | `SUMMARY_API_KEY` | `no-key-required` | API Key |
|  | `SUMMARY_MODEL` | `gemma-4-E4B-it-Q4_K_M` | 模型名 |
|  | `SUMMARY_TEMP` / `SUMMARY_TOP_P` / `SUMMARY_TOP_K` | `1.0` / `0.95` / `64` | 抽取階段取樣參數 |
| Weaviate | `WEAVIATE_HOST` | `localhost` | 主機位址 |
|  | `WEAVIATE_HTTP_PORT` / `WEAVIATE_HTTP_SECURE` | `8080` / `False` | HTTP port / 是否走 TLS |
|  | `WEAVIATE_GRPC_PORT` / `WEAVIATE_GRPC_SECURE` | `50051` / `False` | gRPC port / 是否走 TLS |
| 雙軌去重閘門 | `RAG_IDENTITY_STRONG` | `0.75` | Track A 名字軌強信號門檻 |
|  | `RAG_IDENTITY_KEEP` | `0.62` | Track A 弱信號搭配字面共字才放行 |
|  | `RAG_CONTENT_STRONG` | `0.35` | Track B 描述軌強信號門檻 |
|  | `RAG_CONTENT_MIN` | `0.10` | Track B 配合字面共字才放行的最低門檻 |
| QA Agent | `QA_MAX_CTX_TOKENS` | `65536` | LLM context 上限 (對齊 llama-server 啟動值) |
|  | `QA_CTX_GATE` | `0.7` | 超過此比例觸發 Map-Reduce 篩選 |
|  | `QA_MAX_ITER` | `20` | Tool calling 最大迭代次數 |
|  | `QA_TEMP` / `QA_TOP_P` / `QA_TOP_K` | `1.0` / `0.95` / `64` | QA 階段取樣參數 |

### 執行模式 (`main.py`)

`main.py` 為統一進入點，透過 `--mode` 切換四種流程。

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `--mode` | `ingest` / `process` / `all` / `qa` | `ingest` | 執行模式（見下表） |
| `--novel` | str | `""` | 小說資料夾名稱；`ingest`/`process`/`all` 必填，`qa` 選填（作為偏好作品提示） |
| `--start` | int | `1` | 起始集數 |
| `--vol` | int | `0` | 只跑指定的單一集數（優先於 `--start`） |
| `--clean` | flag | off | 清除該小說的全部輸出後重跑 |
| `--prompt` | str | `""` | QA 模式一次性問題；留空則進入 REPL |
| `--show-graph` | flag | off | QA 模式印出 LangGraph mermaid 圖 |
| `--debug` | flag | off | QA 模式顯示 system prompt、tool 參數與原始檢索輸出 |

模式對照：

- `ingest`：Pre-processing，將原文切為 scene 分片並寫入 `output/<hash>/scenes/`。
- `process`：知識抽取 Agent，跑 LCEL 五階段 pipeline，把 Scene 寫入 `NovelChunk` 並抽取 `NovelEntity`。
- `all`：串跑 `ingest` + `process`。
- `qa`：啟動 QA Agent，支援跨書系 (`scope=all novels`) 或指定作品 (`--novel`) 問答。

### 典型工作流

```bash
# 1. 首次處理一本新小說 (從第 1 卷開始跑完整流程)
uv run python main.py --mode all --novel death_march --start 1

# 2. 補跑單一集 (例如只重處理第 5 卷)
uv run python main.py --mode process --novel death_march --vol 5 --clean

# 3. 啟動跨庫 QA REPL
uv run python main.py --mode qa

# 4. 一次性問答 (帶 debug 看 tool 呼叫細節)
uv run python main.py --mode qa --prompt "聖槍是誰在用？" --debug
```

---

以下為目前的向量庫與儲存架構規劃指南：

## 1. 核心概念 (Core Concepts)

系統採用 **雙層向量架構**：

- **Layer 1 — `NovelChunk`**：敘事分片（scene 粒度），保存原文與摘要兩組向量。作為 RAG 的「事實來源」(ground truth)，即使 Layer 2 條目合併失敗，使用者仍可透過 chunk 向量召回原文，不會漏資訊。
- **Layer 2 — `NovelEntity`**：從 chunk 抽取出的條目（角色、道具、地點等），透過 `chunk_refs` 反指 Layer 1。條目層 merge 失敗從「災難」降級為「索引不漂亮」，不影響檢索。

兩層皆以 **Vol (卷數)** 作為切割單元，建立 **Snapshot Delta (增量快照)** 機制，確保模型在推理特定集數劇情時不會被「未來」的劇透污染。

在資料一致性上，條目的本機檔名統一依賴 **Weaviate 核發的 UUID**，確保同一人物即使別名變化（例如 Saber → 亞瑟王）也不會產生本地 JSON 孤兒檔。Chunk 則以 `(novel_hash, vol_num, scene_index)` 生成 **deterministic UUID (uuid5)**，保證重跑 idempotent。

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

> ⚠️ **Embed 模型限制**：`multilingual-e5-large` 有 512-token 上限，scene 中位數 ~1844 tokens，full_text 向量只覆蓋前 512 tokens。摘要向量（`summary`）用來回補尾段召回。

### 2.2 Layer 2 — `NovelEntity` (條目)

為了方便跨書系搜尋與維護單一索引，該系統將所有小說的條目都集中儲存在 `NovelEntity` 集合中，透過 Metadata 以及自帶的 CJK 分詞器保證精確查找。

### 屬性定義 (Properties)
| 屬性名稱          | 型別 (DataType)       | 說明與防呆機制 |
|-------------------|----------------------|----------------|
| `novel_hash`      | `TEXT`               | 雜湊過的小說編號，做為租戶隔離，查詢時必加 Filter。<br/>*配置 `skip_vectorization=True`，Tokenization `FIELD`* |
| `vol_num`         | `INT`                | 該屬性屬於哪一集的資訊快照設定。 |
| `entity_type`     | `TEXT`               | 受控四類：`character`, `location`, `object`, `concept`。細分標籤推進 `categories`。<br/>*Tokenization `FIELD`（整串視為單一 token，供精準 filter 使用）* |
| `keyword`         | `TEXT`               | 核心 Canonical Name。<br/>*Tokenization `FIELD`，BM25/語義比對交由 `identity` 向量處理* |
| `aliases`         | `TEXT_ARRAY` (陣列)  | 蒐羅所有代稱與方言。<br/>*Tokenization `FIELD`，供 `filter_categories`-類別的精準匹配與 `identity` 向量編碼* |
| `categories`      | `TEXT_ARRAY` (陣列)  | 標籤分類，例如 `["劍", "武器"]`。<br/>*Tokenization `FIELD`，100% 精準匹配「列出所有劍」這類列舉需求* |
| `description`     | `TEXT`               | 豐富的文本描述（含外觀、性格、功能等關鍵設定）。<br/>*Tokenization `GSE` 中文分詞器，供 hybrid BM25 模糊檢索* |
| `appeared_in`     | `INT_ARRAY` (陣列)   | 實體曾在哪些分片（Scene ID）中被提及。由程式碼維護。 |
| `chunk_refs`      | `TEXT_ARRAY` (陣列)  | 指回 Layer 1 (`NovelChunk`) 的 UUID 清單。由程式碼 union 維護。<br/>*配置 `skip_vectorization=True`，Tokenization `FIELD`* |
| `content`         | `TEXT`               | 該實體完整的極簡 JSON 字串備份。<br/>*配置 `skip_vectorization=True`* |
| `tainted`         | `BOOL`               | 失效旗標。為 `True` 時此條目被視為「超級磁鐵」，自動退出 RAG 檢索池與 merge 候選池。<br/>*配置 `skip_vectorization=True`* |
| `tainted_reason`  | `TEXT`               | 失效觸發原因（結構性訊號彙總），供後續 2-pass 工具診斷與重建。<br/>*配置 `skip_vectorization=True`，Tokenization `FIELD`* |

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

> 📌 **為什麼要程式端過濾？** Weaviate hybrid score 是 RRF 融合值，在小資料庫下排第一就近 0.5，與實際語義無關。本協定改以真餘弦相似度把關，並用字面共字做硬閘門，避免語義無關的條目被送進 LLM 誤判合併。

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
6. **失效偵測護欄 (Taint Detection)**：合併完成後，以三組**純結構性/純向量**訊號自動標記「超級磁鐵」條目（例如「清醒夢」吸附了所有技能、魔法、系統事件）：
   - `aliases` 數量超過 `entity_alias_cap` (預設 8) → 別名爆炸
   - `description` 字數超過 `entity_description_cap` (預設 800) → 描述爆炸
   - Merge 當下，新舊 keyword 的 e5 向量 cosine 低於 `entity_semantic_gate` (預設 0.75) → 語意跳躍式合併

   任一觸發即寫入 `tainted=True` 與具體 `tainted_reason`。被標記的條目不再出現在後續 RAG 檢索與 merge 候選池中，停止雪球滾動。**不靠任何特定關鍵字清單**，跨題材/跨語言通用。完整拆分重建交由未來的 2-pass 後處理工具（依 alias 語義聚類拆分 → 重分配 `chunk_refs` → 刪除原 tainted 條目）。

### 3.4 Per-Scene LCEL Pipeline

每個 scene 依序經過五個 Runnable step，以 LCEL 串接：

```
write_chunk_step      → Layer 1 upsert，回傳 chunk_uuid（deterministic UUID，重跑 idempotent）
    │
extract_step          → LLM 抽取 entities
    │
merge_step            → Per-entity：RAG 檢索 → LLM merge/create → inline upsert 到 Layer 2
    │
backfill_chunk_refs   → Scene 結束後，把本場所有 entity UUIDs 回填到 chunk.entity_refs
    │
save_scene_json       → 寫 `world/<type>/<uuid>.json` 與 augment `scene_XXX.json`
```

> 📌 **為什麼 inline upsert？** 每個 entity merge 完成後立即寫入 Weaviate，讓同 scene 後續別名（例如「迪朗達爾 / 聖劍 / 王者之劍」）的 RAG 檢索能看到剛寫入的前置條目，從而正確合併而不是各自被視為新條目。

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
  "filter_type": "可選：character, location, object, concept",
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
   * **運作**：LLM 填入 `filter_type="object", filter_categories=["劍"]`。由於 `categories` 採用 `FIELD` tokenization（整串當一個 token），可直接用 `Filter.contains_any` 做 100% 精準匹配，不會被中文分詞誤切。
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
