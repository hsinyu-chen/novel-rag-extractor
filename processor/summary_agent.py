import json
from typing import Any, Dict, Tuple

from processor.llm_engine import NativeLlamaEngine
from core.prompt_loader import render_prompt


# 卷摘要 JSON Schema（strict mode）—— 由 llama-server 的 response_format 強制格式。
# 只含 LLM 需要 reduce 的語意欄位，pipeline 端會另外附加 updated_scenes。
VOL_SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "theme", "genre", "tone",
        "protagonist", "main_characters",
        "key_locations", "outline", "plot_arcs", "unresolved",
    ],
    "properties": {
        "theme": {"type": "string", "description": "故事主題一句話"},
        "genre": {"type": "string", "description": "類型（奇幻、異世界、科幻…）"},
        "tone": {"type": "string", "description": "敘事語氣（輕鬆、嚴肅、灰暗…）"},
        "protagonist": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "aliases", "titles", "identity", "background"],
            "properties": {
                "name": {"type": "string", "description": "主角慣用本名或暱稱；未知時留空字串"},
                "aliases": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "其他角色對主角的直接稱呼：本名 / 姓 / 字 / 暱稱 / 音譯。不含稱號、職業、偽裝身份、自嘲語。",
                },
                "titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "系統 / 官方 / 族群授予的稱號或封號（凡帶頭銜性質者）。",
                },
                "identity": {"type": "string", "description": "身分 / 職業 / 能力定位（例如：某職業、某種族、某立場）"},
                "background": {"type": "string", "description": "出身 / 世界觀 / 特殊背景"},
            },
        },
        "main_characters": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "role", "note"],
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "description": "夥伴 / 反派 / 導師 / 配角…"},
                    "note": {"type": "string", "description": "與主角的關係或關鍵特徵"},
                },
            },
        },
        "key_locations": {"type": "array", "items": {"type": "string"}},
        "outline": {"type": "string", "description": "累積式劇情大綱，逐場追加"},
        "plot_arcs": {"type": "array", "items": {"type": "string"}},
        "unresolved": {"type": "array", "items": {"type": "string"}},
    },
}


BACKGROUND_COMPACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["background"],
    "properties": {
        "background": {"type": "string", "description": "壓縮後的主角設定敘述，2-5 句話"},
    },
}


def blank_summary() -> Dict[str, Any]:
    return {
        "theme": "",
        "genre": "",
        "tone": "",
        "protagonist": {"name": "", "aliases": [], "titles": [], "identity": "", "background": ""},
        "main_characters": [],
        "key_locations": [],
        "outline": "",
        "plot_arcs": [],
        "unresolved": [],
    }


class SummaryAgent:
    """卷摘要 reducer：
    - update_summary：pass 2（current_summary + scene → updated_summary）
    - compact_background：pass 3（壓縮 protagonist.background 一欄）
    """

    def __init__(self, engine: NativeLlamaEngine):
        self.engine = engine

    def update_summary(
        self,
        current_summary: Dict[str, Any],
        scene_index: int,
        total_scenes: int,
        scene_title: str,
        scene_content: str,
        max_retries: int = 3,
    ) -> Tuple[str, Dict[str, Any], str]:
        """
        回傳 (thought, updated_summary, prompt)。失敗時 updated_summary 回 current_summary（不破壞進度）。
        """
        current_json = json.dumps(current_summary, ensure_ascii=False, indent=2)
        prompt = render_prompt(
            "summary/update_vol_summary",
            current_summary=current_json,
            scene_index=str(scene_index),
            total_scenes=str(total_scenes),
            scene_title=scene_title or "",
            scene_content=scene_content or "",
        )
        messages = [{"role": "user", "content": prompt}]

        last_thought = ""
        for attempt in range(max_retries):
            thought, json_str = self.engine.call_llm(messages, response_schema=VOL_SUMMARY_SCHEMA)
            last_thought = thought or last_thought
            if not json_str:
                continue
            try:
                data = json.loads(json_str)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            return last_thought, data, prompt

        return last_thought, current_summary, prompt

    def compact_background(
        self,
        summary: Dict[str, Any],
        max_retries: int = 3,
    ) -> Tuple[str, str, str]:
        """
        3-pass：把 protagonist.background 重寫成精煉設定敘述。
        回傳 (thought, new_background, prompt)。失敗時 new_background 回原值。
        """
        original = (summary.get("protagonist") or {}).get("background") or ""
        summary_json = json.dumps(summary, ensure_ascii=False, indent=2)
        prompt = render_prompt("summary/compact_background", summary=summary_json)
        messages = [{"role": "user", "content": prompt}]

        last_thought = ""
        for _ in range(max_retries):
            thought, json_str = self.engine.call_llm(messages, response_schema=BACKGROUND_COMPACT_SCHEMA)
            last_thought = thought or last_thought
            if not json_str:
                continue
            try:
                data = json.loads(json_str)
            except Exception:
                continue
            bg = (data or {}).get("background")
            if isinstance(bg, str) and bg.strip():
                return last_thought, bg.strip(), prompt

        return last_thought, original, prompt
