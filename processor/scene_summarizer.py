from typing import List, Dict, Any, Tuple
from processor.llm_engine import NativeLlamaEngine

class SceneSummarizer:
    """
    場景摘要器
    職責：為場景產出極簡短摘要，串接上下文以維持連貫性
    """
    def __init__(self, engine: NativeLlamaEngine):
        self.engine = engine

    def summarize_scene(self, content: str, prev_summary: str = "") -> Tuple[str, str]:
        """
        為單一場景產出極簡短摘要
        """
        messages = [
            {"role": "system", "content": (
                "你的任務是將一段小說場景簡化為一段極簡短的摘要（30-50字）。\n"
                "你會參考『上一個場景的摘要』來確保情節連貫。\n"
                "摘要目標：描述誰、在哪裡、做了什麼核心事件、達成什麼結果（如果有）。\n"
                "請直接輸出摘要，不要有任何前綴（如：摘要：）或額外說明。"
            )},
            {"role": "user", "content": (
                f"【上一個場景摘要】：{prev_summary if prev_summary else '無（這是故事開頭）'}\n\n"
                f"【目前場景內容】：\n{content[:4000]}\n\n"
                "請輸出目前場景的摘要："
            )}
        ]
        raw_text = self.engine.call_llm(messages)
        return self.engine.parse_response(raw_text)
