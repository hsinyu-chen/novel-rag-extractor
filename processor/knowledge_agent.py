import json
import re
from typing import List, Dict, Any, Tuple, Callable
from processor.llm_engine import NativeLlamaEngine

# 禁止的無辨識度 placeholder keyword（以敘述性代號取代）
# - 編號流水系列: 未知人物A/B/1、人物X、角色甲 ...
# - 泛稱單詞: 男子、女子、路人、少年 (獨字時無辨識度)
_PLACEHOLDER_REGEX = re.compile(
    r"^("
    r"(未知人物|人物|角色|未知|無名)[\s_\-]*[A-Za-z0-9甲乙丙丁戊己庚辛壬癸零一二三四五六七八九十]*"
    r"|男子|女子|男人|女人|路人|行人|旁人|某人|少年|少女|老人"
    r")$"
)

class KnowledgeAgent:
    """
    Agentic LLC 工具庫
    負責：提取文本條目、根據 Top-K 候選人決策合併或新建條目。
    具備自動驗證與重試機制。
    """
    def __init__(self, engine: NativeLlamaEngine):
        self.engine = engine

    def _call_with_schema(self, 
                         messages: List[Dict[str, str]], 
                         schema: dict, 
                         validator: Callable[[dict], bool] = None,
                         max_retries: int = 3) -> Tuple[str, dict, list]:
        """
        透過 JSON Schema 保證格式，並加入 3 次重試與自定義邏輯驗證。
        """
        last_thought = ""
        for attempt in range(max_retries):
            # 獲取 LLM 回應
            thought, json_str = self.engine.call_llm(messages, response_schema=schema)
            last_thought = thought
            
            if not json_str:
                print(f"[KnowledgeAgent] Attempt {attempt+1}: Empty response, retrying...")
                continue
                
            try:
                data = json.loads(json_str)
                
                # 執行語意/邏輯驗證
                if validator:
                    if not validator(data):
                        print(f"[KnowledgeAgent] Attempt {attempt+1}: Validation failed, retrying...")
                        continue
                
                return thought, data, messages
            except json.JSONDecodeError as e:
                print(f"[KnowledgeAgent] Attempt {attempt+1}: JSON Parse failed ({str(e)}), retrying...")
                continue
        
        print(f"[KnowledgeAgent] All {max_retries} attempts failed.")
        return last_thought, {}, messages

    def extract_entities(self, scene_content: str, existing_types: List[str] = None) -> Tuple[str, dict, list]:
        """
        LCEL Step 1: 條目提取
        """
        base_types = ["character", "location", "object", "concept"]
        all_types = list(dict.fromkeys(base_types + [t for t in (existing_types or []) if t in base_types]))
        
        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string"},
                            "aliases": {"type": "array", "items": {"type": "string"}},
                            "categories": {"type": "array", "items": {"type": "string"}},
                            "type": {"type": "string", "enum": all_types},
                            "context_summary": {"type": "string"}
                        },
                        "required": ["keyword", "aliases", "categories", "type", "context_summary"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }
        
        types_str = ", ".join(all_types)
        system_prompt = (
            "你是一個專業的小說分析專家。請從文本中提取關鍵條目。準則：\n"
            "1. **禁止提取現實世界事物**：如「日本」、「東京」等。\n"
            "2. **禁止提取「文本夾雜物」(重要)**：爬蟲/掃描/轉載過程中混入的非故事內容，一律不得提取：\n"
            "   - 小說網站/轉載平台的浮水印式署名：如「本文由 XX 論壇轉載」、「輕之國度發布」、「Syosetu 原著」等訊息性字串。\n"
            "   - 譯者/校對/上傳者的署名、後記、前言、版權聲明、催更廣告、贊助訊息、群組 ID。\n"
            "   - 章節標題、卷號、頁碼、分隔線、排版標記等版面元素。\n"
            "   - **判別基準**：該字串**只在文本邊緣出現一次、與劇情角色互動無關**時，才視為夾雜物。\n"
            "   - **反例 (應該提取)**：若小說劇情本身就在描寫出版業、編輯工作、輕小說作者 (如《白箱》《文學少女》題材)，且網站/出版社/編輯職位**作為故事內的實體與角色互動**，則屬於正常條目，請如實提取。\n"
            "3. **使用原文名稱**：若文本已給出官方稱呼或代號，keyword 必須原文照搬。\n"
            "4. **無名人物命名規範 (重要)**：\n"
            "   - **禁止**使用「未知人物A」、「未知人物B」、「人物1」、「角色X」、「男子」、「女人」、「路人」這類無辨識度的編號或泛稱作為 keyword。\n"
            "   - **禁止**使用 A/B/C/甲乙丙 等流水編號，**因為跨場景時你無從得知當前計數位置**，會造成合併錯亂。\n"
            "   - **必做**：用該角色在文本中最具辨識度的**外觀/職能/服裝/種族特徵**組成 2-8 字的敘述性代號。\n"
            "     * 範例：「戴面具的男子」「白髮紅眼少女」「蜥蜴人騎兵長」「穿紅衣的老乞丐」「駕迅猛龍的戰士」。\n"
            "     * 若同一無名角色曾在前文出現過並已經用過某敘述性代號，**沿用該代號**（從描述辨識）。\n"
            "     * 若後續文本揭露真名，直接用真名作 keyword（系統會自動把舊代號納入 aliases）。\n"
            "5. **輸出語言要求**：按照文章原文輸出。\n"
            "6. **type 受控詞彙（只能四選一）**：\n"
            "   - `character`：人物（含 NPC、神祇、擬人化生物）— 有人格、會說話或行動的對象。\n"
            "   - `location`：地點 / 場所 / 世界區域 — 可以「去」的地方（城市、房間、大陸、副本）。\n"
            "   - `object`：實體物品 — 可持有、可摧毀、可交易（武器、裝備、法器、artifact、材料）。\n"
            "   - `concept`：抽象概念 — 技能、魔法、派系、體系、事件、世界設定、規則、知識。\n"
            "7. **人物強制**：任何明顯人名（日文姓名如佐藤、瑞穗，或西方姓名）type 必須為 `character`，嚴禁歸到其他類。\n"
            "8. **過濾瑣碎事物**：禁止提取無獨特性、無劇情重要性的普通日常用品（如：錢包、茶杯、普通衣服）。僅提取具備獨特名稱、特殊能力或對情節有推動作用的關鍵條目。\n"
            "9. **禁止湊數**：如果文本中沒有符合條件的關鍵條目，請返回空陣列 `[]`。嚴禁編造實體或使用「N/A」、「None」、「空白」等佔位符作為關鍵字。\n"
            "10. **type vs categories 區分（重要）**：\n"
            "   - `type`：**四選一** (" + types_str + ")，主幹類別，用於系統分流。\n"
            "   - `categories`：自由標籤陣列，描述此條目的**屬性面向**（題材、派系、職能、用途、風格、子分類等）。**禁止填入 type 值本身**。\n"
            "   - 範例：「黒塚」→ type='character'，categories=['反派','強化人類','暗殺組織']；「聖劍」→ type='object'，categories=['artifact','名劍','雙手武器']；「WAR WORLD」遊戲體系 → type='concept'，categories=['遊戲','虛擬世界','系統']；「風系魔法」→ type='concept'，categories=['magic','屬性','技能']；「王都艾爾迪亞」→ type='location'，categories=['首都','王國','人類領地']。\n"
            "11. **只抽取「世界中的事物」，不抽取「敘事裝置」(重要)**：\n"
            "   - **判準**：若條目在故事世界中作為**一個獨立存在的事物**（具名角色、具名地點、具名物品、具名組織、具名技能/法術/魔法、具名種族、具名事件、具名體系），就抽取。\n"
            "   - **禁止抽取**：\n"
            "     * 讀者視角的敘事/文學手法：伏筆、回憶、轉折、象徵、比喻、旁白。\n"
            "     * 可套用到任意角色的通用心理/感知狀態：如「進入冥想」「產生錯覺」「做夢」「感到恐懼」——除非它是這個世界的**具名現象**（e.g. 設定中明確命名的「冥想術」「夢境網絡」）。\n"
            "     * 通用動作/行為本身：「戰鬥」「對話」「移動」「睡眠」。\n"
            "     * 一般性形容詞或狀態：「高大」「美麗」「平靜」。\n"
            "   - **關鍵提問**：「這個條目**不具名**嗎？或它描述的是**主角做了什麼/處於什麼狀態**而不是**世界裡有一個什麼**？」若是，歸入相關具體 entity 的 `categories` 標籤，不另開 entity。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"【目前場景內容】：\n{scene_content[:4000]}\n\n請提取條目："}
        ]
        
        def validator(data):
            entities = data.get("entities", [])
            if not isinstance(entities, list): return False
            # 硬過濾：過濾掉關鍵字為空、N/A、或違反方案 A 的 placeholder
            valid_entities = []
            for e in entities:
                kw_raw = str(e.get("keyword", "")).strip()
                kw_upper = kw_raw.upper()
                if not kw_raw:
                    continue
                if kw_upper in ["N/A", "NONE", "NULL", "未知", "無"]:
                    continue
                # 禁止編號式或泛稱 placeholder (未知人物A、男子、路人...)，應改用敘述性代號
                if _PLACEHOLDER_REGEX.match(kw_raw):
                    print(f"[KnowledgeAgent] Dropping placeholder keyword: '{kw_raw}' (需使用敘述性代號)")
                    continue
                # 敘述無效則 bypass 整個 keyword (context_summary 為空/N/A/None)
                ctx_raw = str(e.get("context_summary", "")).strip()
                ctx_upper = ctx_raw.upper()
                if not ctx_raw or ctx_upper in ["N/A", "NONE", "NULL", "無", "未知", "無資料", "無相關資訊"]:
                    print(f"[KnowledgeAgent] Dropping keyword '{kw_raw}' due to empty/N/A context_summary")
                    continue
                valid_entities.append(e)
            data["entities"] = valid_entities
            return True

        return self._call_with_schema(messages, schema, validator=validator)

    def create_initial_entity(self, keyword: str, entity_type: str, new_context: str, current_scene_index: int, scene_excerpt: str = "") -> Tuple[str, dict, list]:
        """
        Step 2 - 分流 A: 初始化全然新條目
        """
        schema = {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "categories": {"type": "array", "items": {"type": "string"}},
                "type": {"type": "string"},
                "description": {"type": "string"},
                "major_status_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_index": {"type": "integer"},
                            "event": {"type": "string"}
                        },
                        "required": ["scene_index", "event"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["keyword", "aliases", "categories", "type", "description", "major_status_changes"],
            "additionalProperties": False
        }

        # 根據類型給予描述建議
        desc_advice = ""
        if entity_type == "character":
            desc_advice = "（人物描述應包含：穿著外觀、性格特徵、行動準則等）"
        elif entity_type == "object":
            desc_advice = "（物品描述應包含：外觀材質、用途功能、歷史背景等）"
        elif entity_type == "location":
            desc_advice = "（地點描述應包含：地理外觀、建築風格、文化氛圍、重要性等）"
        elif entity_type == "concept":
            desc_advice = "（概念描述應包含：定義、運作方式、影響範圍、適用情境等）"

        system_prompt = (
            f"你是一個小說知識管理員。請為新發現的 {entity_type} 「{keyword}」建立初始條目。\n"
            f"1. **keyword 核心名稱**：必須強制沿用輸入的名稱「{keyword}」。**嚴禁擅自將其更改為職業、稱號或描述性文字**（範例：禁止將人物名稱改為其職業名稱）。\n"
            f"2. **description 撰寫**：根據情報撰寫豐富的描述。{desc_advice}。**注意：禁止包含任何關於「為什麼這是新條目」的解釋，僅記錄條目內容。**\n"
            f"3. **major_status_changes**：若情節中有重大轉折或初始狀態，請新增一條記錄（scene_index: {current_scene_index}）。\n"
            f"4. **輸出語言要求**：按照文章原文輸出。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"【新條目情報】：\n{new_context}\n\n【場景原文節錄】：\n{scene_excerpt}\n\n請建立條目："}
        ]

        def validator(data):
            # 強制要求關鍵字一致性
            res_kw = str(data.get("keyword", "")).strip()
            if res_kw != keyword:
                print(f"[KnowledgeAgent] Validation failed: Keyword mismatch. Expected '{keyword}', got '{res_kw}'")
                return False
            # 敘述無效則拒絕 (description 為空/N/A)
            desc_raw = str(data.get("description", "")).strip()
            desc_upper = desc_raw.upper()
            if not desc_raw or desc_upper in ["N/A", "NONE", "NULL", "無", "未知", "無資料", "無相關資訊"]:
                print(f"[KnowledgeAgent] Validation failed: Empty/N/A description for '{keyword}'")
                return False
            return True

        return self._call_with_schema(messages, schema, validator=validator)

    def merge_entity(self, keyword: str, entity_type: str, new_context: str, candidates: List[dict], current_scene_index: int, scene_excerpt: str = "") -> Tuple[str, dict, list]:
        """
        Step 2 - 分流 B: 條目合併與情報更新
        """
        schema = {
            "type": "object",
            "properties": {
                "selected_index": {"type": "integer", "description": "0~N 匹配索引，-1 代表皆不匹配 (視為新條目)"},
                "keyword": {"type": "string", "description": "確定的核心名稱"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "categories": {"type": "array", "items": {"type": "string"}},
                "type": {"type": "string"},
                "description": {"type": "string"},
                "major_status_changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_index": {"type": "integer"},
                            "event": {"type": "string"}
                        },
                        "required": ["scene_index", "event"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["selected_index", "keyword", "aliases", "categories", "type", "description", "major_status_changes"],
            "additionalProperties": False
        }

        candidate_text = ""
        for i, cand in enumerate(candidates):
            candidate_text += f"候選 [{i}]:\n{json.dumps(cand, ensure_ascii=False)}\n\n"

        # 根據類型給予描述建議
        desc_advice = ""
        if entity_type == "character":
            desc_advice = "（人物描述應包含：穿著外觀、性格特徵、行動準則等）"
        elif entity_type == "object":
            desc_advice = "（物品描述應包含：外觀材質、用途功能、歷史背景等）"
        elif entity_type == "location":
            desc_advice = "（地點描述應包含：地理外觀、建築風格、文化氛圍、重要性等）"
        elif entity_type == "concept":
            desc_advice = "（概念描述應包含：定義、運作方式、影響範圍、適用情境等）"

        system_prompt = (
            f"你是一個小說知識管理員。請判斷新情報是否對應候選名單中的某個 {entity_type}。\n"
            f"\n"
            f"⚠️ **核心判斷原則 — 先讀再答**：\n"
            f"RAG 檢索只是「相似候選」，不等於「確定是同一個條目」。判斷基準如下：\n"
            f"  A. **預設值為 -1 (視為新條目)**。只有在明確滿足下列任一條件時才選擇匹配索引：\n"
            f"     - 候選的 `keyword` 或 `aliases` 與輸入名稱「{keyword}」**字面高度重合**（例如：輸入「亞瑟王」對候選「阿爾托莉雅」有已記錄的別名連結；或輸入「劍」對候選「魔劍」有共同漢字「劍」）。\n"
            f"     - **或** 候選描述中有與新情報**直接對應的同一個物件/同一個人**的明確線索（例如相同獨特特徵、相同事件）。\n"
            f"  B. **以下情況必須填 -1，嚴禁匹配**：\n"
            f"     - 輸入名稱與候選名稱**類別完全不同**（例：輸入「劍」是武器，候選「儲倉」是系統）。\n"
            f"     - 只是**場景背景相同**（同一遊戲、同一事件、同一地點提及），但指涉的實體不同。\n"
            f"     - 候選描述談論的是**另一個類別**的條目（例：輸入是「item / 劍」，候選描述是「程式設計師的工作」）。\n"
            f"  C. **RAG 順從陷阱**：即使候選清單只有一兩個選項，若都不對應就必須填 -1，禁止「將就匹配」。\n"
            f"\n"
            f"1. **selected_index**：依上述原則，匹配填索引 (0~N)；否則填 -1。\n"
            f"2. **keyword 鎖定與更新**：如果匹配且候選人已有正式名稱（非空、非 N/A、且非「未知人物」），則必須沿用該 keyword。**但若候選人名稱無效（為空或 N/A）或為「未知人物」，則必須改用新情報中的精確名稱「{keyword}」。**\n"
            f"3. **description 撰寫**：整合新舊情報，撰寫豐富的描述。{desc_advice}。**注意：禁止在此欄位中包含任何關於「為什麼匹配」或「為什麼不對應」的判斷理由與邏輯，該欄位應僅包含對條目本身的描述內容。**\n"
            f"4. **major_status_changes**：**只輸出本 scene ({current_scene_index}) 新觀察到的狀態變更**，不要重複列出候選清單中已記錄過的舊項目（系統會自動合併）。若本 scene 沒有新的重大轉折、受傷、升級或損毀，請輸出空陣列 `[]`。同一個 scene 最多一條，禁止為同一事件產生多筆描述。\n"
            f"5. **aliases 規範**：只有當候選確實是同一實體時，才將「{keyword}」加入 aliases；**若 selected_index = -1，aliases 必須是全新的陣列**，不得把「{keyword}」放進去（它本身就是 keyword）。\n"
            f"6. **輸出語言要求**：按照文章原文輸出。\n"
            f"7. **不匹配處理 (-1)**：若 selected_index 為 -1，必須強制沿用輸入名稱「{keyword}」作為 keyword，嚴禁將其改為職業、稱號或一般性描述文字。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"【候選清單】：\n{candidate_text}\n【新情報】：\n{new_context}\n【場景原文節錄】：\n{scene_excerpt}\n\n請決策："}
        ]

        def validator(data):
            idx = data.get("selected_index")
            if idx is None or not (-1 <= idx < len(candidates)): return False
            res_kw = str(data.get("keyword", "")).strip()
            
            # 嚴格驗證關鍵字
            if idx == -1:
                if res_kw != keyword:
                    print(f"[KnowledgeAgent] Validation failed: New entity keyword mismatch. Expected '{keyword}', got '{res_kw}'")
                    return False
            
            if not res_kw or res_kw.upper() in ["N/A", "NONE", "NULL"]: return False
            # 敘述無效則拒絕 (description 為空/N/A)
            desc_raw = str(data.get("description", "")).strip()
            desc_upper = desc_raw.upper()
            if not desc_raw or desc_upper in ["N/A", "NONE", "NULL", "無", "未知", "無資料", "無相關資訊"]:
                print(f"[KnowledgeAgent] Validation failed: Empty/N/A description for '{keyword}'")
                return False
            return True

        return self._call_with_schema(messages, schema, validator=validator)
