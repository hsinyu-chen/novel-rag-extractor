import operator
import re
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


def _strip_tool_call_tokens(text: str) -> str:
    """兜底：Gemma chat template 偶爾把 tool_call 的特殊 token 原樣落進 content，
    清掉 <|tool_call>...<tool_call|> / <tool_call>...</tool_call> 之類殘留。"""
    if not text:
        return text
    patterns = [
        r"<\|?tool_call\|?>.*?<\|?/?tool_call\|?>",   # <|tool_call>...<tool_call|> 等變體
        r"<\|tool_call\|>.*?<\|/tool_call\|>",
        r"<tool_call>.*?</tool_call>",
    ]
    out = text
    for p in patterns:
        out = re.sub(p, "", out, flags=re.DOTALL)
    return out.strip()


class QueryAgent:
    """
    LangGraph FSM 查詢代理。

    流程：
      plan (LLM+tools) ──tool_calls──► tool ──► take_notes ──► plan
         │
         └──無 tool_calls / 強制切換──► answer ──► END

    強制切換條件：token_ratio >= ctx_gate（預設 0.7），或 iteration >= max_iter。

    所有節點共用同一份 system prompt（由 initial_state 指定，預設 DEFAULT_SYSTEM）。
    answer 節點不沿用 plan 的 messages，而是只消費 `notes`（take_notes 整理過的
    工具結果摘要）+ 原始問題，並走未綁工具的 LLM 產最終答案 —
    避免 tool_call 配對問題與原始 JSON 膨脹 prompt。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        tools: List[BaseTool],
        tokenize_fn=None,
        max_ctx_tokens: int = 65536,
        ctx_gate: float = 0.7,
        max_iter: int = 20,
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
        # 把整理好的 notes 包裝成一對合成的 tool_call / tool_result 餵給未綁工具的 self.llm，
        # 讓 LLM 以「檢索工具剛剛回傳這批事實」的訓練分佈來消費，而不是把 notes 當成
        # 使用者提供的素材（避免『根據您提供的資料…』這類語氣偏差）。
        # 結尾再補一則 HumanMessage redirect → 把模型從「tool-loop」情境拉回
        # 「使用者要答案」情境，避免 Gemma chat template 把對話尾端是 ToolMessage
        # 誤讀成「該再發 tool_call」而吐出 <|tool_call> 原始模板 token。
        system_msg = next(
            (m for m in state["messages"] if isinstance(m, SystemMessage)),
            SystemMessage(content=DEFAULT_SYSTEM),
        )
        notes = state.get("notes", []) or []
        notes_block = "\n\n".join(notes) if notes else "（本次未進行工具檢索。）"

        tool_call_id = "retrieved_notes_bundle"
        synthetic_ai = AIMessage(
            content="",
            tool_calls=[{
                "id": tool_call_id,
                "name": "retrieved_notes",
                "args": {"question": state["question"]},
                "type": "tool_call",
            }],
        )
        synthetic_tool = ToolMessage(
            tool_call_id=tool_call_id,
            name="retrieved_notes",
            content=notes_block,
        )
        redirect = HumanMessage(content=(
            "以上是檢索工具回傳的事實。請直接依 system prompt 規則，以自然語言輸出對原始問題"
            f"「{state['question']}」的最終答案，**不要再呼叫任何工具、不要輸出 <tool_call> 之類的模板標記**。"
        ))

        ai = self.llm.invoke([
            system_msg,
            HumanMessage(content=state["question"]),
            synthetic_ai,
            synthetic_tool,
            redirect,
        ])
        content = _strip_tool_call_tokens(ai.content or "")
        return {"final_answer": content}

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
    def initial_state(
        self,
        question: str,
        system: Optional[str] = None,
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> QAState:
        """
        建立一次 graph run 的初始 state。

        history_messages 應已包含前幾輪的 `[HumanMsg_q, synthetic_ai(tool_call), synthetic_tool(notes), AIMsg_ans]`
        自然對話流；本方法不再做特別的歷史 notes 重打包。每輪 `state["notes"]` 只收集本輪新產生的 notes。
        """
        system_msg = SystemMessage(content=system or DEFAULT_SYSTEM)
        user_msg = HumanMessage(content=question)

        msgs: List[BaseMessage] = [system_msg]
        if history_messages:
            msgs.extend(history_messages)
        msgs.append(user_msg)

        return {
            "question": question,
            "messages": msgs,
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
