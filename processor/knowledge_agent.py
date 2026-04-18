import json
from typing import List, Dict, Any, Tuple, Callable
from processor.llm_engine import NativeLlamaEngine

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
                         max_retries: int = 3) -> Tuple[str, dict]:
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
                
                return thought, data
            except json.JSONDecodeError as e:
                print(f"[KnowledgeAgent] Attempt {attempt+1}: JSON Parse failed ({str(e)}), retrying...")
                continue
        
        print(f"[KnowledgeAgent] All {max_retries} attempts failed.")
        return last_thought, {}

    def extract_entities(self, scene_content: str, existing_types: List[str] = None) -> Tuple[str, dict]:
        """
        LCEL Step 1: 條目提取
        """
        base_types = ["character", "item", "poi", "faction", "magic", "skill", "event", "artifact", "world-setting"]
        all_types = list(set(base_types + (existing_types or [])))
        
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
            "2. **使用原文名稱**：Keyword 必須是小說文本中的官方稱呼或代號。\n"
            "3. **人物命名**：有名用名，無名用「未知人物A」、「未知人物B」。\n"
            "4. **繁體中文**：所有輸出內容必須使用正體/繁體中文。\n"
            "5. **分類參考**：盡量使用現有類別：" + types_str + "\n"
            "6. **人物強制**：如果是人物，必須為 'character'。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"【目前場景內容】：\n{scene_content[:4000]}\n\n請提取條目："}
        ]
        
        def validator(data):
            return isinstance(data.get("entities"), list)

        return self._call_with_schema(messages, schema, validator=validator)

    def create_initial_entity(self, keyword: str, entity_type: str, new_context: str, current_scene_index: int) -> Tuple[str, dict]:
        """
        Step 2 - 分流 A: 初始化全然新條目 (極簡結構)
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
        elif entity_type in ["item", "artifact", "magic"]:
            desc_advice = "（物品描述應包含：外觀材質、用途功能、歷史背景等）"
        elif entity_type in ["poi", "world-setting"]:
            desc_advice = "（地點/設定描述應包含：地理外觀、建築風格、文化氛圍等）"

        system_prompt = (
            f"你是一個小說知識管理員。請為新發現的 {entity_type} 「{keyword}」建立初始條目。\n"
            f"1. **description 撰寫**：根據情報撰寫豐富的描述。{desc_advice}\n"
            f"2. **major_status_changes**：若情節中有重大轉折或初始狀態，請新增一條記錄（scene_index: {current_scene_index}）。\n"
            f"3. **輸出語言要求**：按照文章原文輸出。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"【新條目情報】：\n{new_context}\n\n請建立條目："}
        ]

        return self._call_with_schema(messages, schema)

    def merge_entity(self, keyword: str, entity_type: str, new_context: str, candidates: List[dict], current_scene_index: int) -> Tuple[str, dict]:
        """
        Step 2 - 分流 B: 條目合併與情報更新 (極簡結構)
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
            candidate_text += f"候選人 [{i}]:\n{json.dumps(cand, ensure_ascii=False)}\n\n"

        # 根據類型給予描述建議
        desc_advice = ""
        if entity_type == "character":
            desc_advice = "（人物描述應包含：穿著外觀、性格特徵、行動準則等）"
        elif entity_type in ["item", "artifact", "magic"]:
            desc_advice = "（物品描述應包含：外觀材質、用途功能、歷史背景等）"
        elif entity_type in ["poi", "world-setting"]:
            desc_advice = "（地點/設定描述應包含：地理外觀、建築風格、文化氛圍等）"

        system_prompt = (
            f"你是一個小說知識管理員。請判斷新情報是否對應候選名單中的某個 {entity_type}。\n"
            f"1. **selected_index**：匹配填索引，若皆不匹配（即使 RAG 找出了相似對象，但你判斷不是）則填 -1。\n"
            f"2. **keyword 鎖定**：如果匹配，除非候選人原名為「未知人物」，否則必須沿用候選人的 keyword。\n"
            f"3. **description 撰寫**：整合新舊情報，撰寫豐富的描述。{desc_advice}\n"
            f"4. **major_status_changes**：若新情報中有重大轉折、受傷、升級、損毀等，請新增一條記錄（scene_index: {current_scene_index}）。\n"
            f"5. **繁體中文要求**：全繁體輸出。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"【候選清單】：\n{candidate_text}\n【新情報】：\n{new_context}\n\n請決策："}
        ]

        def validator(data):
            idx = data.get("selected_index")
            if not (-1 <= idx < len(candidates)): return False
            return True

        return self._call_with_schema(messages, schema, validator=validator)
