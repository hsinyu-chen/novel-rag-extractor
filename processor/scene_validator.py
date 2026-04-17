import json
from typing import List, Dict, Any, Tuple, Optional, Callable
from llama_cpp import Llama as NativeLlama
from processor.llm_engine import NativeLlamaEngine

class SceneValidator:
    """
    場景驗證器
    職責：判斷兩段文字是否屬於同一個「敘事段落」
    """
    def __init__(self, engine: NativeLlamaEngine):
        self.engine = engine

    def validate_boundaries(self, scenes: List[str], on_boundary_checked: Optional[Callable] = None, on_scene_ready: Optional[Callable] = None) -> List[str]:
        # 移除空值或純空白的片段
        scenes = [s for s in scenes if s.strip()]
        
        if len(scenes) <= 1:
            if on_scene_ready and len(scenes) == 1:
                on_scene_ready(1, scenes[0])
            return scenes

        validated_scenes: List[str] = []
        current_scene = scenes[0]
        total_boundaries = len(scenes) - 1

        for i in range(1, len(scenes)):
            next_scene = scenes[i]
            s1_end = current_scene[-400:] if len(current_scene) > 400 else current_scene
            s2_begin = next_scene[:400] if len(next_scene) > 400 else next_scene

            messages = [
                {"role": "system", "content": (
                    "你是專業的小說結構分析師。你的任務是判斷兩段文字是否屬於同一個「敘事段落」。\n"
                    "以下任一條件成立，就應該分開（combine: false）：\n"
                    "- 時間跳躍（即使很短，如「過了一會」「隔天」）\n"
                    "- 地點轉換\n"
                    "- 視角或焦點人物切換\n"
                    "- 話題或事件焦點明顯改變（例如從戰鬥轉為對話、從行動轉為回憶）\n"
                    "- 情緒或敘事節奏出現轉折（例如從緊張轉為平靜）\n"
                    "- 出現分隔符號（如 ＊＊＊、───、空行分段）\n"
                    "只有當兩段文字在描述完全相同的連續場景、沒有任何上述轉折時，才合併。\n"
                    "只輸出 JSON，不要附加說明。"
                )},
                {"role": "user", "content": (
                    f"【片段 1 結尾】：\n{s1_end}\n\n"
                    f"【片段 2 開頭】：\n{s2_begin}\n\n"
                    "請判斷：{\"combine\": true} 或 {\"combine\": false}"
                )}
            ]

            raw_text = self.engine.call_llm(messages)

            # --- 解析結果 ---
            try:
                thought, json_str = self.engine.parse_response(raw_text)
                data = json.loads(json_str)
                should_combine = data.get("combine", True)
            except Exception:
                should_combine = "true" in raw_text.lower() if "combine" in raw_text.lower() else True
                thought = "Parse Error"

            if should_combine:
                current_scene = current_scene + "\n" + next_scene
            else:
                validated_scenes.append(current_scene)
                if on_scene_ready:
                    on_scene_ready(len(validated_scenes), current_scene)
                current_scene = next_scene

            if on_boundary_checked:
                on_boundary_checked(i, total_boundaries, len(validated_scenes) + 1, thought[:50] if thought else "No COT")

        validated_scenes.append(current_scene)
        if on_scene_ready:
            on_scene_ready(len(validated_scenes), current_scene)
        return validated_scenes
