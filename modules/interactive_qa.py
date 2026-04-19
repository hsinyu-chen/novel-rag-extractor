import os
import hashlib
from typing import Any, List

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage

from processor.query_agent import QueryAgent, DEFAULT_SYSTEM
from processor.query_tool import build_query_tools
from processor.weaviate_storage import WeaviateStorage
from processor.embed_engine import LlamaSimpleEmbeddings


class InteractiveQA:
    """
    小說 QA Agent 的 CLI 入口。支援 REPL 互動與一次性 --prompt 模式。
    """

    def __init__(
        self,
        weaviate_db: WeaviateStorage,
        embed_engine: LlamaSimpleEmbeddings,
        config: Any,
    ):
        self.weaviate_db = weaviate_db
        self.embed_engine = embed_engine
        self.conf = config
        self.console = Console()

    # ---------- 工具 ----------
    def _get_path_hash(self, path: str) -> str:
        return hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]

    def _detect_max_vol(self, novel_hash: str, override: int = 0) -> int:
        """若使用者沒指定 vol，掃 output/{hash}/world/ 底下最大的 vol_N。"""
        if override and override > 0:
            return override
        world_dir = os.path.join("output", novel_hash, "world")
        if not os.path.isdir(world_dir):
            return 1
        max_vol = 0
        for name in os.listdir(world_dir):
            if name.startswith("vol_"):
                try:
                    v = int(name.split("_", 1)[1])
                    max_vol = max(max_vol, v)
                except Exception:
                    continue
        return max_vol or 1

    def _build_system_prompt(self, novel_name: str, novel_hash: str, max_vol: int) -> str:
        """從 Weaviate 拉本書統計 + 全 DB 收錄狀況，拼進 system prompt。"""
        db_novels = []
        try:
            db_novels = self.weaviate_db.list_novels()
        except Exception as e:
            print(f"[InteractiveQA] list_novels failed: {e}")

        profile = {}
        try:
            profile = self.weaviate_db.get_novel_profile(novel_hash)
        except Exception as e:
            print(f"[InteractiveQA] get_novel_profile failed: {e}")

        # --- 全庫概況 ---
        db_lines = [f"【資料庫收錄】共 {len(db_novels)} 部小說："]
        for n in db_novels:
            h = n.get("novel_hash", "")
            vols = n.get("vols") or []
            vol_nums = [v["vol_num"] for v in vols]
            marker = "  ← 目前會話" if h == novel_hash else ""
            db_lines.append(
                f"  - hash={h} | vols={vol_nums or '(無)'} | "
                f"scenes={n.get('chunk_count', 0)} | entities={n.get('entity_count', 0)}{marker}"
            )

        # --- 本書詳細 ---
        vols = profile.get("vols") or []
        vol_detail = "、".join([f"vol{v['vol_num']}({v['scene_count']}scenes)" for v in vols]) or "(無)"
        type_map = profile.get("entity_by_type") or {}
        type_str = "、".join([f"{k}:{v}" for k, v in type_map.items()]) or "(無)"
        top = profile.get("top_entities") or []
        top_str = "、".join([f"{e['keyword']}({e['type']},{e['appearances']}場)" for e in top]) or "(無)"

        cur_lines = [
            f"\n【目前會話小說】{novel_name}  (hash={novel_hash})",
            f"  卷分佈：{vol_detail}",
            f"  可查詢上限卷數：{max_vol}（vol_num > {max_vol} 的資料會被過濾）",
            f"  條目總數：{profile.get('entity_count', 0)}  → {type_str}",
            f"  高出場條目：{top_str}",
        ]

        context_block = "\n".join(db_lines + cur_lines)
        return context_block + "\n\n" + DEFAULT_SYSTEM

    def _build_agent(self, novel_hash: str, max_vol: int) -> QueryAgent:
        tools = build_query_tools(self.weaviate_db, novel_hash=novel_hash, max_vol=max_vol)
        # 用 SUMMARY_* 這組 config 當作 QA LLM 入口（同一台 llama-server）
        return QueryAgent(
            base_url=self.conf.get("summary_base_url"),
            api_key=self.conf.get("summary_api_key"),
            model=self.conf.get("summary_model"),
            tools=tools,
            tokenize_fn=self.embed_engine.tokenize,
            max_ctx_tokens=int(self.conf.get("qa_max_ctx_tokens", 8192)),
            ctx_gate=float(self.conf.get("qa_ctx_gate", 0.7)),
            max_iter=int(self.conf.get("qa_max_iter", 8)),
            temperature=float(self.conf.get("qa_temp", 1.0)),
            top_p=float(self.conf.get("qa_top_p", 0.95)),
            top_k=int(self.conf.get("qa_top_k", 64)),
        )

    # ---------- 渲染 ----------
    def _render_stream(
        self,
        agent: QueryAgent,
        question: str,
        system_prompt: str = None,
        history_messages: List[BaseMessage] = None,
        prior_notes: List[str] = None,
        debug: bool = False,
    ):
        state = agent.initial_state(question, system=system_prompt)
        if history_messages:
            # 將過往 Q&A 插在 system 與本輪問題之間，供 plan 節點看到脈絡
            state["messages"] = [state["messages"][0]] + list(history_messages) + [state["messages"][1]]
        if prior_notes:
            state["notes"] = list(prior_notes)

        last_msg_count = len(state["messages"])
        notes_so_far: List[str] = list(prior_notes or [])
        final_answer = ""
        final_notes: List[str] = list(notes_so_far)
        thinking_printed = False

        self.console.rule(f"[bold cyan]Question[/bold cyan]")
        self.console.print(question)
        self.console.rule()

        if not debug:
            self.console.print("[dim]agent 正在思考...[/dim]")
            thinking_printed = True

        for event in agent.graph.stream(state, stream_mode="values"):
            msgs = event.get("messages", [])
            ratio = event.get("token_ratio", 0.0)
            iteration = event.get("iteration", 0)
            final_answer = event.get("final_answer", "") or final_answer
            final_notes = event.get("notes", final_notes) or final_notes

            # 只打印新出現的訊息
            new_msgs = msgs[last_msg_count:]
            last_msg_count = len(msgs)

            for m in new_msgs:
                if isinstance(m, AIMessage):
                    tcs = m.tool_calls or []
                    if tcs:
                        for tc in tcs:
                            name = tc.get("name")
                            args = tc.get("args", {})
                            if debug:
                                self.console.print(
                                    f"[bold yellow][plan #{iteration}][/bold yellow] "
                                    f"[green]-> {name}[/green]({self._fmt_args(args)})  "
                                    f"[dim]ctx {ratio:.0%}[/dim]"
                                )
                            else:
                                self.console.print(f"[dim]agent 正在查詢 {name}...[/dim]")
                    else:
                        if debug:
                            self.console.print(
                                f"[bold yellow][plan #{iteration}][/bold yellow] "
                                f"[dim]no tool_calls → 進入 answer 模式 (ctx {ratio:.0%})[/dim]"
                            )
                        else:
                            self.console.print("[dim]agent 正在整理回答...[/dim]")
                elif isinstance(m, ToolMessage):
                    if debug:
                        body = (m.content or "").strip()
                        preview = body[:180] + ("..." if len(body) > 180 else "")
                        self.console.print(f"  [blue][tool:{m.name}][/blue] {preview}")

            new_notes = event.get("notes", [])
            if debug and len(new_notes) > len(notes_so_far):
                added = new_notes[len(notes_so_far):]
                for note in added:
                    self.console.print(f"  [magenta][note saved][/magenta] ({len(note)} chars)")
            notes_so_far = new_notes or notes_so_far

        self.console.rule("[bold green]Answer[/bold green]")
        if final_answer:
            try:
                self.console.print(Panel(Markdown(final_answer), border_style="green"))
            except Exception:
                self.console.print(final_answer)
        else:
            self.console.print("[red](no answer produced)[/red]")
        self.console.rule()

        return final_answer, final_notes

    def _fmt_args(self, args: dict) -> str:
        parts = []
        for k, v in (args or {}).items():
            if v in (None, "", [], 0):
                continue
            parts.append(f"{k}={v!r}")
        return ", ".join(parts)

    # ---------- 對外入口 ----------
    def run(
        self,
        novel_name: str,
        prompt: str = "",
        vol: int = 0,
        show_graph: bool = False,
        debug: bool = False,
    ):
        base_data_path = os.path.join("data", novel_name)
        novel_hash = self._get_path_hash(base_data_path)
        max_vol = self._detect_max_vol(novel_hash, override=vol)

        self.console.print(
            f"[bold magenta]>> QA Agent[/bold magenta]  "
            f"novel=[cyan]{novel_name}[/cyan]  hash=[cyan]{novel_hash}[/cyan]  "
            f"max_vol=[cyan]{max_vol}[/cyan]"
        )

        agent = self._build_agent(novel_hash, max_vol)
        system_prompt = self._build_system_prompt(novel_name, novel_hash, max_vol)

        if show_graph:
            self.console.print("[dim]--- Graph (mermaid) ---[/dim]")
            self.console.print(agent.export_mermaid())
            self.console.print("[dim]-----------------------[/dim]")

        if debug:
            self.console.print(Panel(system_prompt, title="[dim]system prompt[/dim]", border_style="dim"))

        if prompt:
            try:
                self._render_stream(agent, prompt, system_prompt=system_prompt, debug=debug)
            finally:
                try:
                    self.weaviate_db.close()
                except Exception:
                    pass
            return

        # REPL（連續問答：保留前幾輪 Q&A 與累積的 notes 作為下一輪 context）
        history_messages: List[BaseMessage] = []
        accumulated_notes: List[str] = []
        self.console.print("[dim]輸入問題後 Enter；輸入 'exit' 或 Ctrl+C 結束；':reset' 清空對話紀錄。[/dim]")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]bye.[/dim]")
                return
            if not q:
                continue
            if q.lower() in ("exit", "quit", ":q"):
                self.console.print("[dim]bye.[/dim]")
                return
            if q.lower() == ":reset":
                history_messages = []
                accumulated_notes = []
                self.console.print("[dim](對話紀錄已清空)[/dim]")
                continue
            try:
                final_answer, final_notes = self._render_stream(
                    agent,
                    q,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    prior_notes=accumulated_notes,
                    debug=debug,
                )
                history_messages.append(HumanMessage(content=q))
                history_messages.append(AIMessage(content=final_answer or ""))
                accumulated_notes = final_notes
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")
