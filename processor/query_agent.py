import operator
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from core.prompt_loader import load_prompt


class QAState(TypedDict):
    question: str
    messages: Annotated[List[BaseMessage], add_messages]
    notes: Annotated[List[str], operator.add]
    iteration: int
    token_ratio: float
    final_answer: str


DEFAULT_SYSTEM = load_prompt("query/agent_system")
ANSWER_SYSTEM = load_prompt("query/answer_system")


class QueryAgent:
    """
    LangGraph FSM 查詢代理。

    流程：
      plan (LLM+tools) ──tool_calls──► tool ──► take_notes ──► plan
         │
         └──無 tool_calls / 強制切換──► answer ──► END

    強制切換條件：token_ratio >= ctx_gate（預設 0.7），或 iteration >= max_iter。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        tools: List[BaseTool],
        tokenize_fn=None,
        max_ctx_tokens: int = 8192,
        ctx_gate: float = 0.7,
        max_iter: int = 8,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 0,
    ):
        self.tools = tools
        self.tokenize_fn = tokenize_fn
        self.max_ctx_tokens = max_ctx_tokens
        self.ctx_gate = ctx_gate
        self.max_iter = max_iter

        # llama-server 經 OpenAI-compatible 介面；強制 enable_thinking=True 以讓 Gemma 走 CoT
        extra_body = {"chat_template_kwargs": {"enable_thinking": True}}
        if top_k and top_k > 0:
            extra_body["top_k"] = top_k
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
        )
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.llm_answer = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            extra_body=extra_body,
        )
        self.graph = self._build_graph()

    # ---------- Token 估算 ----------
    def _estimate_tokens(self, messages: List[BaseMessage], notes: List[str]) -> int:
        if self.tokenize_fn is None:
            # Fallback: 粗略以字元數 / 2 估算（CJK 約 1:1，英 / 符號偏小）
            total_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
            total_chars += sum(len(n) for n in (notes or []))
            return total_chars // 2
        total = 0
        for m in messages:
            content = getattr(m, "content", "") or ""
            if content:
                total += self.tokenize_fn(content)
            tool_calls = getattr(m, "tool_calls", None) or []
            for tc in tool_calls:
                total += self.tokenize_fn(str(tc.get("args", "")))
        for n in (notes or []):
            total += self.tokenize_fn(n)
        return total

    # ---------- Nodes ----------
    def _plan_node(self, state: QAState) -> dict:
        msgs = state["messages"]
        ai: AIMessage = self.llm_with_tools.invoke(msgs)
        tokens = self._estimate_tokens(msgs + [ai], state.get("notes", []))
        ratio = min(1.0, tokens / self.max_ctx_tokens)
        return {
            "messages": [ai],
            "iteration": state.get("iteration", 0) + 1,
            "token_ratio": ratio,
        }

    def _tool_node(self, state: QAState) -> dict:
        # 直接借用 prebuilt ToolNode 處理 tool_calls
        tn = ToolNode(self.tools)
        return tn.invoke(state)

    def _take_notes_node(self, state: QAState) -> dict:
        """把最新一批 ToolMessage 的內容擷取為 notes，避免原始 JSON 被塞進 answer prompt。"""
        new_notes = []
        for m in reversed(state["messages"]):
            if isinstance(m, ToolMessage):
                content = m.content if isinstance(m.content, str) else str(m.content)
                new_notes.append(f"[{m.name}] {content}")
            else:
                break
        new_notes.reverse()
        return {"notes": new_notes}

    def _answer_node(self, state: QAState) -> dict:
        clean_msgs: List[BaseMessage] = [
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(
                content=(
                    f"【使用者問題】\n{state['question']}\n\n"
                    f"【資料庫查詢結果】\n" + "\n\n".join(state.get("notes") or ["（無查詢結果）"])
                )
            ),
        ]
        ai = self.llm_answer.invoke(clean_msgs)
        return {"final_answer": ai.content or ""}

    # ---------- Routing ----------
    def _route_after_plan(self, state: QAState) -> Literal["tool", "answer"]:
        last = state["messages"][-1] if state["messages"] else None
        over_budget = state.get("token_ratio", 0.0) >= self.ctx_gate
        over_iter = state.get("iteration", 0) >= self.max_iter
        has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
        if over_budget or over_iter:
            return "answer"
        if has_tool_calls:
            return "tool"
        return "answer"

    # ---------- Graph ----------
    def _build_graph(self):
        g = StateGraph(QAState)
        g.add_node("plan", self._plan_node)
        g.add_node("tool", self._tool_node)
        g.add_node("take_notes", self._take_notes_node)
        g.add_node("answer", self._answer_node)
        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", self._route_after_plan, {
            "tool": "tool",
            "answer": "answer",
        })
        g.add_edge("tool", "take_notes")
        g.add_edge("take_notes", "plan")
        g.add_edge("answer", END)
        return g.compile()

    # ---------- Public ----------
    def initial_state(self, question: str, system: Optional[str] = None) -> QAState:
        system_msg = SystemMessage(content=system or DEFAULT_SYSTEM)
        user_msg = HumanMessage(content=question)
        return {
            "question": question,
            "messages": [system_msg, user_msg],
            "notes": [],
            "iteration": 0,
            "token_ratio": 0.0,
            "final_answer": "",
        }

    def export_mermaid(self) -> str:
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            return f"(mermaid render failed: {e})"
