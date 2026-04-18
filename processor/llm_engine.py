import re
import json
from typing import List, Dict, Any, Tuple
from openai import OpenAI

class NativeLlamaEngine:
    """
    原生 Llama 引擎 (OpenAI API Compatible)
    職責：管理模型生命週期，提供統一的聊天與解析接口，原生支援 json_schema 結構化輸出與從 reasoning_content 解析 CoT
    """
    def __init__(self, base_url: str, api_key: str, model: str, params: Dict[str, Any]):
        print(f"Connecting to OpenAI API at {base_url} (model: {model})...")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.params = params

    def call_llm(self, messages: List[Dict[str, str]], response_schema: dict = None) -> Tuple[str, str]:
        """
        統一的 LLM 調用入口
        回傳：(thought, content)
        """
        call_params = dict(
            model=self.model,
            messages=messages,
            temperature=self.params.get("temperature", 1.0),
            top_p=self.params.get("top_p", 0.95),
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": True
                }
            }
        )

        if response_schema:
            call_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": response_schema
                }
            }

        response = self.client.chat.completions.create(**call_params)
        message = response.choices[0].message
        
        content = message.content or ""
        # 優先從 API 原生欄位取得 reasoning (CoT)
        thought = getattr(message, "reasoning_content", None) or ""
        
        # Fallback: 如果 API 沒給但 content 裡面有 <think> 標籤（部分 proxy 或舊版相容性）
        if not thought:
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                thought = think_match.group(1).strip()
                content = content[think_match.end():].strip()
        
        # 如果是 JSON 模式，進一步確保內容純淨
        if response_schema:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]

        return thought, content
