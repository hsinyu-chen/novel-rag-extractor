from typing import List, Dict, Any, Tuple
from processor.llm_engine import NativeLlamaEngine
from core.prompt_loader import load_prompt

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
            {"role": "system", "content": load_prompt("scene/summarize_scene")},
            {"role": "user", "content": (
                f"【上一個場景摘要】：{prev_summary if prev_summary else '無（這是故事開頭）'}\n\n"
                f"【目前場景內容】：\n{content[:4000]}\n\n"
                "請輸出目前場景的摘要："
            )}
        ]
        thought, summary = self.engine.call_llm(messages)
        return thought, summary
