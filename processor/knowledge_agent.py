import json
from typing import List, Dict, Any, Tuple
from processor.llm_engine import NativeLlamaEngine

class KnowledgeAgent:
    """
    Agentic LLC 工具庫
    負責：提取文本條目、與現有條目進行合併 (基於 JSON Schema)
    """
    def __init__(self, engine: NativeLlamaEngine):
        self.engine = engine

    def _call_with_schema(self, messages: List[Dict[str, str]], schema: dict) -> Tuple[str, dict]:
        """
        透過 JSON Schema 保證格式正確，不需重試機制
        """
        thought, json_str = self.engine.call_llm(messages, response_schema=schema)
        
        if not json_str:
            return thought, {}
            
        try:
            parsed_data = json.loads(json_str)
            return thought, parsed_data
        except json.JSONDecodeError as e:
            print(f"[KnowledgeAgent] JSON Parse failed: {str(e)}")
            return thought, {}

    def extract_entities(self, scene_content: str, existing_types: List[str] = None) -> Tuple[str, dict]:
        """
        LCEL Step 1: 條目提取
        """
        type_description = "請依常理自行判斷英文單詞 (例如: character, item, poi, faction, magic, spell 等)。重要：若是個人物，請務必填寫 'character'"
        if existing_types:
            type_description = f"請盡量從資料庫已存的類型中挑選適合的 ({', '.join(existing_types)})，若都不適合再自行發明新類別。若是常規人物請填寫 'character'"

        schema = {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "keyword": {"type": "string", "description": "條目名稱"},
                            "aliases": {"type": "array", "items": {"type": "string"}, "description": "別名、代稱或簡稱。若無請留空陣列"},
                            "categories": {"type": "array", "items": {"type": "string"}, "description": "條目標籤。如人物可寫 ['種族', '職業']，武器可寫 ['劍']"},
                            "type": {"type": "string", "description": type_description},
                            "context_summary": {"type": "string", "description": "對該條目在此場景中的表現或描述總結"}
                        },
                        "required": ["keyword", "aliases", "categories", "type", "context_summary"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["entities"],
            "additionalProperties": False
        }
        
        messages = [
            {"role": "system", "content": (
                "你是一個專業的小說分析專家。你的任務是從這段劇情中萃取出所有有意義的條目，包含人物、地點、物品、世界設定等。\n"
                "注意：如果這段沒有條目，請回傳空的 entities 陣列：{\"entities\": []}"
            )},
            {"role": "user", "content": f"【目前場景內容】：\n{scene_content[:4000]}\n\n請提取該片段的人物及關鍵條目，並以 JSON 輸出："}
        ]
        return self._call_with_schema(messages, schema)

    def merge_character(self, keyword: str, new_context: str, existing_data: dict, current_scene_index: int) -> Tuple[str, dict]:
        """
        專屬於 Character 的合併邏輯
        """
        schema = {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "categories": {"type": "array", "items": {"type": "string"}},
                "type": {"type": "string", "enum": ["character"]},
                "profile": {
                    "type": "object",
                    "properties": {
                        "action_principles": {"type": "string"},
                        "personality": {"type": "string"},
                        "appearance": {"type": "string"},
                        "clothing": {"type": "string"},
                        "equipment": {"type": "string"}
                    },
                    "required": ["action_principles", "personality", "appearance", "clothing", "equipment"],
                    "additionalProperties": False
                },
                "last_known_location": {"type": "string"},
                "turning_points": {
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
            "required": ["keyword", "aliases", "categories", "type", "profile", "last_known_location", "turning_points"],
            "additionalProperties": False
        }
        
        system_prompt = (
            "你是一個人物檔案管理員。你需要將「新場景中獲得的人物情報」合併進「現存的人物檔案」中。\n"
            "你必須維護並輸出一份完整的角色 JSON 檔案，確保豐富各項欄位與保留舊有資料。"
        )

        user_content = f"【現存檔案】：\n{json.dumps(existing_data, ensure_ascii=False) if existing_data else '完全沒有，這是一個新角色。'}\n\n"
        user_content += f"【新場景情報 (Scene {current_scene_index})】：\n{new_context}\n\n"
        user_content += "請進行資訊融合，如果新情報中有角色的重點行為/變化，請在 turning_points 陣列新增一筆。若沒有變化，則維持原 turning_points。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        return self._call_with_schema(messages, schema)

    def merge_generic_entity(self, keyword: str, entity_type: str, new_context: str, existing_data: dict) -> Tuple[str, dict]:
        """
        針對 Item, Poi, World-Setting 的通用合併邏輯
        """
        schema = {
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "aliases": {"type": "array", "items": {"type": "string"}},
                "categories": {"type": "array", "items": {"type": "string"}},
                "type": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["keyword", "aliases", "categories", "type", "description"],
            "additionalProperties": False
        }
        
        system_prompt = (
            f"你是一個專案設定管理員。需要將「新獲得的情報」合併進「現存的 {entity_type} 檔案」中。\n"
            "請盡可能豐富並保留原有資訊，並回傳格式化 JSON。"
        )

        user_content = f"【現存檔案】：\n{json.dumps(existing_data, ensure_ascii=False) if existing_data else '無，這是新條目。'}\n\n"
        user_content += f"【新情報】：\n{new_context}\n\n"
        user_content += "請結合上述兩者，總結成最完整的 description。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        return self._call_with_schema(messages, schema)
