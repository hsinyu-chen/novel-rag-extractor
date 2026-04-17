import re
import json
from typing import List, Dict, Any, Tuple
from llama_cpp import Llama as NativeLlama

class NativeLlamaEngine:
    """
    原生 Llama 引擎 (GGUF 自動偵測)
    職責：管理模型生命週期，提供統一的聊天與解析接口
    """
    def __init__(self, model_path: str, params: Dict[str, Any]):
        print(f"Loading Native Llama Model from {model_path}...")
        self.client = NativeLlama(
            model_path=model_path,
            n_ctx=params.get("n_ctx", 8192),
            n_gpu_layers=params.get("n_gpu_layers", -1),
            verbose=False,
            embedding=False
        )
        self.params = params

        # 取得 GGUF 自動偵測的 Jinja2 handler
        self._handler = self.client._chat_handlers.get(self.client.chat_format)
        thinking_status = "ENABLED" if self._handler else "DISABLED (no Jinja2 template)"
        print(f"  * Chat format: {self.client.chat_format}")
        print(f"  * Native Thinking: {thinking_status}")

    def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        統一的 LLM 調用入口
        """
        call_params = dict(
            messages=messages,
            temperature=self.params.get("temperature", 1.0),
            top_p=self.params.get("top_p", 0.95),
            top_k=self.params.get("top_k", 64),
        )

        if self._handler:
            response = self._handler(
                llama=self.client,
                **call_params,
                enable_thinking=True,
            )
        else:
            response = self.client.create_chat_completion(**call_params)

        return response["choices"][0]["message"]["content"]

    @staticmethod
    def parse_response(raw_text: str) -> Tuple[str, str]:
        """
        解析 thinking + answer
        """
        thought = ""
        answer = raw_text

        channel_split = re.split(r"<channel\|>", raw_text, maxsplit=1)
        if len(channel_split) == 2:
            thought = re.sub(r"<\|channel>thought\n?", "", channel_split[0]).strip()
            answer = channel_split[1].strip()
        else:
            think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
            if think_match:
                thought = think_match.group(1).strip()
                answer = raw_text[think_match.end():].strip()

        # 從 answer 部分提取 JSON (如果有的話)
        json_match = re.search(r"\{[^}]*\}", answer)
        if json_match:
            return thought, json_match.group()

        return thought, answer
