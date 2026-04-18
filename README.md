# Narrative RAG Pipeline - VectorDB 架構設計

本專案利用 **Weaviate (v4)** 向量資料庫與 **Gemma 4** 語言模型，打造具備「條目去重」、「分卷快照功能」以及「多重命名向量」的小說 RAG Agent 系統。

以下為目前的向量庫與儲存架構規劃指南：

## 1. 核心概念 (Core Concepts)

我們將條目資料（角色、道具、地點等）的狀態以 **Vol (卷數)** 作為切割單元，建立起 **Snapshot Delta (增量快照)** 的機制，確保模型在後續推理特定集數的劇情時，不會被來自「未來」的劇透與設定給污染。

在資料一致性上，我們捨棄使用「條目名稱 (Keyword)」作為本機條目 JSON 檔名的方式。條目的本機檔名統一依賴 **Weaviate 核發的 UUID**。這徹底解決了同一個人物在劇中因為被更改別名、職階（例如：從 Saber 變成 亞瑟王）而導致本地 JSON 孤兒檔案分歧的痛點。

---

## 2. Weaviate Collection 實作結構 (`NovelEntity`)

為了方便跨書系搜尋與維護單一索引，我們將所有小說的條目都集中儲存在 `NovelEntity` 集合中，透過 Metadata 以及自帶的 CJK 分詞器保證精確查找。

### 屬性定義 (Properties)
| 屬性名稱          | 型別 (DataType)       | 說明與防呆機制 |
|-------------------|----------------------|----------------|
| `novel_hash`      | `TEXT`               | 雜湊過的小說編號 (例如 `f9b8afed`)，做為租戶隔離，查詢時必加 Filter。<br/>*配置 `skip_vectorization=True`* |
| `vol_num`         | `INT`                | 該屬性屬於哪一集的資訊快照設定。 |
| `entity_type`     | `TEXT`               | 例如 `character`, `item`, `poi`, `world-setting`。<br/>*使用 `Tokenization.GSE` 中文分詞器* |
| `keyword`         | `TEXT`               | 最常用的指標性稱呼。<br/>*使用 `Tokenization.GSE` 中文分詞器* |
| `aliases`         | `TEXT_ARRAY` (陣列)  | **這是解決同音/代稱分裂的核心**。蒐羅所有代稱與方言。<br/>*使用 `Tokenization.GSE` 中文分詞器* |
| `categories`      | `TEXT_ARRAY` (陣列)  | 條目分類標籤（例如：`["劍", "武器"]`、`["人類", "劍士"]`）。方便大範圍類別檢索。<br/>*使用 `Tokenization.GSE` 中文分詞器* |
| `appeared_in`     | `INT_ARRAY` (陣列)   | 所有的出場紀錄，精確記錄這個條目曾在哪些分片（Scene ID）中被提及。 |
| `content`         | `TEXT`               | `KnowledgeAgent` 返回的該條目完整 JSON 字串備份。<br/>*配置 `skip_vectorization=True`* |

> 💡 **中文分詞技術細節**：
> 包含 `aliases` 在內的所有可搜尋文字皆掛上了 **GSE (Go Segmenter)** 分詞器，解決了 Weaviate 中文長字串被視為單一 Token 的缺陷。

---

## 3. 多重命名向量與檢索策略 (Named Vectors - BYOV)

專案採用 **Bring Your Own Vectors (BYOV)** 設計，模型使用 `multilingual-e5-large` 進行精準編碼（確保已加裝 `passage:` 前綴）。Weaviate 內部維護三組獨立的向量空間：

1. **`identity` 向量**：利用 `keyword` 與 `aliases` 計算。專門用來找人。
2. **`equipment` 向量**：利用人物的 `profile.equipment`（或道具自身的描述）計算。使得未來使用者搜尋「EX咖哩棒」時，能透過給予 `equipment` 向量高權重，精確地叫出它的持有者。
3. **`context` 向量**：利用完整的個性、外觀與長文描述計算，專門對付「那個綁馬尾藍眼睛的女孩是誰」此類模糊語義提問。

### 新情報合併時的搜索策略：Hybrid Search
當從新場景抽出一段條目描述時，在新建條目前，系統會使用 Weaviate 的 **Hybrid Search (Alpha = 0.5)** 去撞擊 `context` 向量與 BM25 關鍵字：
* 如果別名完全命中，BM25 權重發威保證提取到現存檔案。
* 如果別名寫錯但描述相似，e5 Vector 發威把檔案拉回來。
* 只有當真的找不到時，才會建立全新條目。

---

## 4. 本機磁碟的 JSON 備份協定

除了 VectorDB 的紀錄，為了後續還原、Debug 與其他 RAG 第二層系統運用，我們同樣維護一份最高高精度的本地 JSON 關聯。

#### 條目檔案 (Entity Profiles)
路徑：`output/<hash>/world/vol_<X>/<type>/<UUID>.json`
每一次 Weaviate 更新，就同步用最新的 JSON 內容將檔案複寫在本卷的資料夾下。同一卷內重複出現的人物會不斷堆疊 `appeared_in` 與 `turning_points` 資訊。

#### 場景分片附加資訊 (Scene Enhancements)
路徑：`output/<hash>/scenes/processed/vol_<X>/scene_<Y>.json`
在每個分片的處理尾聲，該分片會補上自己內部出現過的所有條目清單：
```json
{
  ...
  "entities_extracted": [
     {
       "keyword": "阿爾托莉雅",
       "type": "character",
       "uuid": "8a92b23-...."
     }
  ]
}
```
這個架構將允許未來的「Scene Agent（分片特化 RAG 總結器）」能夠 `O(1)` 精準帶著 UUID 去本機抓取當時完整的條目世界觀，徹底消除因為別名不同導致找不到伏筆與裝備的錯誤。

---

## 5. 檢索架構規劃 (RAG Agent 查詢藍圖)

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
   * **運作**：LLM 填入 `filter_type="item", filter_categories=["劍"]`。由於 `categories` 掛載了 GSE 中文分詞，立刻 100% 取回對應物件。
3. **統計列舉功能 (List & Sort Fetching)**
   * **情境**：「請列出這本小說裡出場次數最高的主要人物前五名。」
   * **運作**：LLM 取巧判斷不需語意，傳入 `query_text="", filter_type="character", sort_by="appearances", limit=5`。Python 後端偵測到文字留空，**不走 Vector**，直接化身傳統 Database 叫出資料，並以 `appeared_in` 的涵蓋廣度作為主角排名指標，輕鬆解決大模型的計數弱點。

---

## 6. 後處理與 Context 排序策略 (RAG Post-Retrieval)

### 1. 嚴格時序排列 (Chronological/Causal Sorting)
在組裝 RAG Prompt 時，**嚴格棄用任何破壞時序的重排演算法** (如 LongContextReorder)。
無論是萃取角色生平歷史或檢索連續劇情分片，一律**依照 Vol / Scene Index 發展的先後順序進行線性排列**送入大語言模型，以此來激發模型最佳的「狀態變化與因果推理」能力。

### 2. 棄用傳統 Reranker (防範語意破壞)
絕大部份市面上的輕量級 Rerank 模型（如 `bge-reranker`）皆基於百科全書或標準問答集訓練，其設計邏輯偏向事實檢索。對於小說中極端複雜的「隱喻」、「伏筆」、「人物心境」與「文學性描述」理解能力極差。
若強悍介入這類泛用型 Reranker 進行二次打分，極容易發生**「干擾大於效果」**的慘況——把沒有直接命中關鍵字、但實際上是核心伏筆的情境分片當成雜訊剃除。
因此，只要檢索回的 Context 總量座落在 64K Token 的防守範圍內，本系統將完全信任 Weaviate Hybrid Search (`e5` 語意 + `BM25` 精準) 的初篩結果，一律無碼直通給 Gemma-4 處理，保證最深層的文學解讀不被破壞。

### 3. 超量 Context 應對：FSM 多輪篩選代理 (Map-Reduce)
若遇到諸如「總結整部小說所有大事件」這類注定突破 64K 物理上限的史詩級檢索，我們也不依賴粗暴的截斷或 Rerank。
取而代之的是啟動 **FSM (有限狀態機) Agent** 進行多段式處理：
* **Map (分塊過篩)**：將破表的 Context 依照卷數或事件鍊切分為多個安全的批次，LLM 在這幾個平行或序列狀態中，執行「是否有命中問題意圖」的獨立初篩。
* **Reduce (狀態收斂)**：將各輪過篩保留下來的高純度結點交給最終的狀態機進行整合推理。以此機制打破硬體 Context 極限，實現真正意義上的無損全本分析。
