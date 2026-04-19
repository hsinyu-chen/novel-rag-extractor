import json
import re
from typing import List, Dict, Any, Tuple, Callable
from processor.llm_engine import NativeLlamaEngine
from core.prompt_loader import render_prompt

# 依 entity_type 決定 description 撰寫建議（原為 if/elif 分支，抽成查表）
_DESC_ADVICE = {
    "character": "（人物描述應包含：穿著外觀、性格特徵、行動準則等）",
    "object": "（物品描述應包含：外觀材質、用途功能、歷史背景等）",
    "location": "（地點描述應包含：地理外觀、建築風格、文化氛圍、重要性等）",
    "concept": "（概念描述應包含：定義、運作方式、影響範圍、適用情境等）",
}

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
        system_prompt = render_prompt("extraction/extract_entities", types_str=types_str)

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

        system_prompt = render_prompt(
            "extraction/create_initial_entity",
            entity_type=entity_type,
            keyword=keyword,
            desc_advice=_DESC_ADVICE.get(entity_type, ""),
            current_scene_index=current_scene_index,
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

        system_prompt = render_prompt(
            "extraction/merge_entity",
            entity_type=entity_type,
            keyword=keyword,
            desc_advice=_DESC_ADVICE.get(entity_type, ""),
            current_scene_index=current_scene_index,
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
