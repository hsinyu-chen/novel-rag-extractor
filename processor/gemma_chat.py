import json
from typing import Any, List, Optional, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import OpenAI
from pydantic import Field


def _to_openai_msg(m: BaseMessage) -> dict:
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": m.content or ""}
    if isinstance(m, HumanMessage):
        return {"role": "user", "content": m.content or ""}
    if isinstance(m, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": getattr(m, "tool_call_id", "") or "",
            "content": m.content if isinstance(m.content, str) else str(m.content),
        }
    if isinstance(m, AIMessage):
        d: dict = {"role": "assistant", "content": m.content or ""}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": tc.get("name") or "",
                        "arguments": json.dumps(tc.get("args") or {}, ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ]
        return d
    return {"role": "user", "content": str(getattr(m, "content", ""))}


class GemmaThinkingChat(BaseChatModel):
    """
    薄殼 LangChain BaseChatModel，直用 openai SDK 呼叫 llama-server，
    把 Gemma 的 reasoning_content 透傳到 AIMessage.additional_kwargs，
    保留 LangGraph / bind_tools 的使用界面。
    """

    base_url: str
    api_key: str
    model: str
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 0
    tools: List[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "gemma-thinking-chat"

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        **kwargs: Any,
    ) -> "GemmaThinkingChat":
        return self.model_copy(update={"tools": list(tools)})

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        extra_body: dict = {"chat_template_kwargs": {"enable_thinking": True}}
        if self.top_k and self.top_k > 0:
            extra_body["top_k"] = self.top_k

        payload: dict = {
            "model": self.model,
            "messages": [_to_openai_msg(m) for m in messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "extra_body": extra_body,
        }
        if self.tools:
            payload["tools"] = [convert_to_openai_tool(t) for t in self.tools]

        resp = client.chat.completions.create(**payload)
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        tool_calls: List[dict] = []
        raw_tcs = getattr(msg, "tool_calls", None) or []
        for tc in raw_tcs:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": args,
                "type": "tool_call",
            })

        additional: dict = {}
        if reasoning:
            additional["reasoning_content"] = reasoning

        ai = AIMessage(
            content=content,
            additional_kwargs=additional,
            tool_calls=tool_calls,
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])
