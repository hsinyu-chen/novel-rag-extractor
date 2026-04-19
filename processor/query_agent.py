import operator
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
)
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from core.prompt_loader import load_prompt
from processor.gemma_chat import GemmaThinkingChat


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

        # 直連 openai SDK（跟 llm_engine.py 同一套），確保 enable_thinking 生效
        # 且把 Gemma 的 reasoning_content 透傳進 AIMessage.additional_kwargs。
        self.llm = GemmaThinkingChat(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        self.llm_with_tools = self.llm.bind_tools(tools)
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
        # 沿用 plan 階段的 messages（含原 system + 全部 tool 呼叫 / 結果），
        # 僅在尾巴追加 Answer Mode 指令，最大化 llama-server KV-cache 前綴重用。
        # 若 LLM 仍嘗試呼叫工具，以 ToolMessage 駁回後重試（最多 3 次）。
        msgs: List[BaseMessage] = list(state["messages"])

        # Edge case：token/iter 上限切換 answer 時，plan 的最後一則 AIMessage 可能還掛著
        # 未執行的 tool_calls，屬於 OpenAI 無效狀態；補合成 ToolMessage 收尾。
        last = msgs[-1] if msgs else None
        if isinstance(last, AIMessage) and last.tool_calls:
            for tc in last.tool_calls:
                msgs.append(ToolMessage(
                    tool_call_id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    content="Answer Mode：已達 token/iteration 上限，本次檢索略過。",
                ))

        directive = ANSWER_SYSTEM + (
            f"\n\n---\n【使用者原始問題】\n{state['question']}\n\n"
            "請立刻依上述規則輸出對這個問題的最終答案文字，不要回覆「已切換模式」之類的確認語。"
        )
        msgs.append(HumanMessage(content=directive))

        for attempt in range(3):
            ai = self.llm_with_tools.invoke(msgs)
            tool_calls = getattr(ai, "tool_calls", None) or []
            if not tool_calls:
                return {"final_answer": ai.content or ""}

            # LLM 違反 Answer Mode，仍輸出 tool_calls → 合規駁回並要求直接回答
            msgs.append(ai)
            for tc in tool_calls:
                msgs.append(ToolMessage(
                    tool_call_id=tc.get("id", ""),
                    name=tc.get("name", ""),
                    content="Answer Mode：工具呼叫被拒。請勿再呼叫工具，直接輸出最終答案文字。",
                ))
            msgs.append(HumanMessage(content=(
                f"上述工具呼叫已被系統駁回（Answer Mode 禁止呼叫工具，重試 {attempt + 1}/3）。"
                f"請立刻改輸出針對原始問題「{state['question']}」的純文字答案，"
                "不要再包含任何 tool_calls。"
            )))

        return {"final_answer": "(answer mode: LLM 反覆嘗試呼叫工具，已中止)"}

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
