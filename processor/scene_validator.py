import os
import json
from typing import List, Optional, Callable

from processor.llm_engine import NativeLlamaEngine
from core.prompt_loader import load_prompt


class SceneValidator:
    """
    場景驗證器
    職責：判斷兩段文字是否屬於同一個「敘事段落」
    """
    def __init__(self, engine: NativeLlamaEngine):
        self.engine = engine

    def validate_boundaries(
        self,
        scenes: List[str],
        on_boundary_checked: Optional[Callable] = None,
        on_scene_ready: Optional[Callable] = None,
        log_dir: Optional[str] = None,
        max_tokens: int = 0,
        min_tokens: int = 128,
        tokenizer: Optional[Callable[[str], int]] = None,
    ) -> List[str]:
        """
        Args:
            max_tokens: 單一場景的 token 上限。超過就強制切斷，不問 LLM。0 = 不限制。
            min_tokens: 小於此值的碎片會被合併到前一個場景。0 = 不清理。
            tokenizer: 計算 token 數的 callback，例如 lambda text: len(engine.tokenize(text))
        """
        # 移除空值或純空白的片段
        scenes = [s for s in scenes if s.strip()]

        if len(scenes) <= 1:
            if on_scene_ready and len(scenes) == 1:
                on_scene_ready(1, scenes[0])
            return scenes

        # 準備 log 目錄
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        validated_scenes: List[str] = []
        current_scene = scenes[0]
        total_boundaries = len(scenes) - 1

        for i in range(1, len(scenes)):
            next_scene = scenes[i]

            # --- Token 上限硬切 ---
            if max_tokens > 0 and tokenizer:
                merged_tokens = tokenizer(current_scene + "\n" + next_scene)
                if merged_tokens > max_tokens:
                    # 超過上限，強制切斷，跳過 LLM 判斷
                    validated_scenes.append(current_scene)
                    if on_scene_ready:
                        on_scene_ready(len(validated_scenes), current_scene)
                    current_scene = next_scene

                    if log_dir:
                        log_entry = {
                            "boundary": f"{i}/{total_boundaries}",
                            "decision": "FORCE_SPLIT",
                            "reason": f"merged_tokens={merged_tokens} > max_tokens={max_tokens}",
                        }
                        log_path = os.path.join(log_dir, f"boundary_{i:03d}.json")
                        with open(log_path, "w", encoding="utf-8") as f:
                            json.dump(log_entry, f, ensure_ascii=False, indent=2)

                    if on_boundary_checked:
                        on_boundary_checked(i, total_boundaries, len(validated_scenes) + 1, f"FORCE_SPLIT ({merged_tokens}>{max_tokens})")
                    continue

            # --- LLM 判斷 ---
            s1_end = current_scene[-400:] if len(current_scene) > 400 else current_scene
            s2_begin = next_scene[:400] if len(next_scene) > 400 else next_scene
            
            # 使用 JSON Schema 確保邊界判斷格式
            schema = {
                "type": "object",
                "properties": {
                    "combine": {"type": "boolean"}
                },
                "required": ["combine"],
                "additionalProperties": False
            }

            messages = [
                {"role": "system", "content": load_prompt("scene/validate_boundaries")},
                {"role": "user", "content": (
                    f"【片段 1 結尾】：\n{s1_end}\n\n"
                    f"【片段 2 開頭】：\n{s2_begin}\n\n"
                    "請判斷：{\"combine\": true} 或 {\"combine\": false}"
                )}
            ]

            thought, json_str = self.engine.call_llm(messages, response_schema=schema)

            # --- 解析結果 ---
            try:
                data = json.loads(json_str)
                should_combine = data.get("combine", True)
            except Exception:
                should_combine = "true" in json_str.lower() if "combine" in json_str.lower() else True
                thought = thought or "Parse Error"

            # --- Debug Log ---
            if log_dir:
                log_entry = {
                    "boundary": f"{i}/{total_boundaries}",
                    "prompt": {
                        "system": messages[0]["content"],
                        "user": messages[1]["content"],
                    },
                    "response": {
                        "thought": thought,
                        "json_str": json_str,
                        "parsed_result": should_combine,
                    },
                }
                log_path = os.path.join(log_dir, f"boundary_{i:03d}.json")
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log_entry, f, ensure_ascii=False, indent=2)

            if should_combine:
                current_scene = current_scene + "\n" + next_scene
            else:
                validated_scenes.append(current_scene)
                current_scene = next_scene

            if on_boundary_checked:
                on_boundary_checked(i, total_boundaries, len(validated_scenes) + 1, thought[:50] if thought else "No COT")

        validated_scenes.append(current_scene)

        # --- 碎片清理：小於 min_tokens 的 scene 合併到前一個 ---
        if min_tokens > 0 and tokenizer and len(validated_scenes) > 1:
            merged: List[str] = [validated_scenes[0]]
            for scene in validated_scenes[1:]:
                if tokenizer(scene) < min_tokens:
                    merged[-1] = merged[-1] + "\n" + scene
                else:
                    merged.append(scene)
            # 第一個 scene 如果也太小，合併到下一個
            if len(merged) > 1 and tokenizer(merged[0]) < min_tokens:
                merged[1] = merged[0] + "\n" + merged[1]
                merged.pop(0)
            validated_scenes = merged

        # 通知最終結果
        if on_scene_ready:
            for idx, scene in enumerate(validated_scenes, 1):
                on_scene_ready(idx, scene)

        return validated_scenes
