import os
import hashlib
import json
from typing import Any, Dict, List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from processor.json_storage import JsonStorage
from processor.weaviate_storage import WeaviateStorage
from processor.summary_agent import SummaryAgent, blank_summary


class SummaryProcess:
    """
    卷摘要 pass 2 + pass 3 管線：
      - Pass 2 (`update_summary`)：逐 scene 讀原文，每場呼叫 LLM 直接 reduce 出結構化卷摘要 JSON。
      - Pass 3 (`compact_background`)：pass 2 跑完後額外重寫 `protagonist.background`，避免逐場累積成編年史。

    - 狀態檔：output/{novel_hash}/summary/vol_{N}.json（含 updated_scenes 與 compacted 旗標，可斷點續跑）
    - 來源：從 Weaviate NovelChunk fetch（全文保證與 ingest 一致）
    """

    def __init__(
        self,
        storage: JsonStorage,
        weaviate_db: WeaviateStorage,
        agent: SummaryAgent,
        config: Any,
    ):
        self.storage = storage
        self.weaviate_db = weaviate_db
        self.agent = agent
        self.conf = config
        self.console = Console()

    def get_path_hash(self, path: str) -> str:
        return hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:8]

    # ---------- 狀態檔 I/O ----------
    def _summary_key(self, novel_hash: str, vol_num: int) -> str:
        return f"{novel_hash}.summary.vol_{vol_num}"

    def _load_state(self, novel_hash: str, vol_num: int) -> Dict[str, Any]:
        data = self.storage.get(self._summary_key(novel_hash, vol_num)) or {}
        summary = data.get("summary") or blank_summary()
        updated = list(data.get("updated_scenes") or [])
        compacted = bool(data.get("compacted"))
        return {"summary": summary, "updated_scenes": updated, "compacted": compacted}

    def _save_state(self, novel_hash: str, vol_num: int, summary: Dict[str, Any], updated_scenes: List[int], compacted: bool = False):
        payload = {
            "novel_hash": novel_hash,
            "vol_num": vol_num,
            "updated_scenes": sorted(set(updated_scenes)),
            "compacted": compacted,
            "summary": summary,
        }
        self.storage.set(self._summary_key(novel_hash, vol_num), payload)

    def _save_log(self, novel_hash: str, vol_num: int, scene_idx: int, payload: Dict[str, Any]):
        log_dir = os.path.join("output", novel_hash, "logs", "summary", f"vol_{vol_num}")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"scene_{scene_idx:03d}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.console.print(f"[yellow]summary log save failed: {e}[/yellow]")

    # ---------- Scene 來源 ----------
    def _list_scenes(self, novel_hash: str, vol_num: int) -> List[Dict[str, Any]]:
        """
        從 Weaviate 列出指定卷的所有 scene（含 title / content / token_count），依 scene_index 排序。
        """
        collection = self.weaviate_db._client.collections.get("NovelChunk")
        from weaviate.classes.query import Filter
        flt = Filter.by_property("novel_hash").equal(novel_hash) & \
              Filter.by_property("vol_num").equal(int(vol_num))
        try:
            resp = collection.query.fetch_objects(filters=flt, limit=1000)
        except Exception as e:
            self.console.print(f"[red]Failed to list scenes: {e}[/red]")
            return []
        rows = []
        for obj in resp.objects:
            p = obj.properties
            rows.append({
                "scene_index": int(p.get("scene_index") or 0),
                "title": p.get("title", "") or "",
                "content": p.get("content", "") or "",
                "token_count": int(p.get("token_count") or 0),
            })
        rows.sort(key=lambda x: x["scene_index"])
        return rows

    # ---------- 主流程 ----------
    def run(
        self,
        novel_name: str,
        start_vol: int = 1,
        end_vol: int = 0,
        clean_output: bool = False,
    ):
        base_data_path = os.path.join("data", novel_name)
        if not os.path.isdir(base_data_path):
            self.console.print(f"[red]Error: 找不到小說資料夾 {base_data_path}[/red]")
            return

        novel_hash = self.get_path_hash(base_data_path)
        self.console.print(f"\n[bold magenta]>> 開始卷摘要 2-pass: {novel_name}[/bold magenta] (Hash: {novel_hash})")

        # 找出所有 vol（透過 Weaviate 的分卷 aggregate）
        profile = self.weaviate_db.get_novel_profile(novel_hash)
        vol_infos = profile.get("vols") or []
        if not vol_infos:
            self.console.print(f"[red]Weaviate 內沒有 {novel_name} 的 scene 資料；請先跑 ingest + process。[/red]")
            return

        vol_nums = [v["vol_num"] for v in vol_infos
                    if start_vol <= v["vol_num"] and (end_vol == 0 or v["vol_num"] <= end_vol)]
        vol_nums.sort()

        for v in vol_nums:
            self._run_vol(novel_hash, v, clean_output)

        self.console.print(f"\n[bold green]{novel_name} 卷摘要完成。[/bold green]")

    def _run_vol(self, novel_hash: str, vol_num: int, clean_output: bool):
        self.console.print(f"\n[cyan]>> Vol {vol_num} 摘要中...[/cyan]")

        if clean_output:
            # 重跑時清掉該卷的摘要狀態與 log
            key_path = os.path.join("output", novel_hash, "summary", f"vol_{vol_num}.json")
            if os.path.exists(key_path):
                os.remove(key_path)
            log_dir = os.path.join("output", novel_hash, "logs", "summary", f"vol_{vol_num}")
            if os.path.isdir(log_dir):
                import shutil
                shutil.rmtree(log_dir)

        state = self._load_state(novel_hash, vol_num)
        summary = state["summary"]
        updated = set(state["updated_scenes"])
        compacted = state["compacted"]

        scenes = self._list_scenes(novel_hash, vol_num)
        total = len(scenes)
        if total == 0:
            self.console.print(f"[yellow]Vol {vol_num} 沒有 scene，略過。[/yellow]")
            return

        pending = [s for s in scenes if s["scene_index"] not in updated]
        self.console.print(f"  [dim]{total} 個 scene；待處理 {len(pending)} 個（已處理 {len(updated)}）[/dim]")

        if not pending:
            summary, compacted = self._maybe_compact(novel_hash, vol_num, summary, compacted)
            self._save_state(novel_hash, vol_num, summary, sorted(updated), compacted=compacted)
            return

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), console=self.console,
        ) as progress:
            task = progress.add_task(f"[bold blue]Vol {vol_num}", total=len(pending))
            for scene in pending:
                idx = scene["scene_index"]
                progress.update(task, description=f"[blue]Vol {vol_num} scene {idx}/{total}")
                try:
                    thought, new_summary, prompt = self.agent.update_summary(
                        current_summary=summary,
                        scene_index=idx,
                        total_scenes=total,
                        scene_title=scene["title"],
                        scene_content=scene["content"],
                    )
                except Exception as e:
                    self.console.print(f"[red]Vol {vol_num} scene {idx} failed: {e}[/red]")
                    progress.update(task, advance=1)
                    continue

                self._save_log(novel_hash, vol_num, idx, {
                    "scene_index": idx,
                    "title": scene["title"],
                    "token_count": scene["token_count"],
                    "prompt": prompt,
                    "thought": thought,
                    "summary_in": summary,
                    "summary_out": new_summary,
                })

                summary = new_summary
                updated.add(idx)
                self._save_state(novel_hash, vol_num, summary, sorted(updated), compacted=False)
                progress.update(task, advance=1)

        if len(updated) >= total:
            summary, compacted = self._maybe_compact(novel_hash, vol_num, summary, compacted)
            self._save_state(novel_hash, vol_num, summary, sorted(updated), compacted=compacted)

    def _maybe_compact(self, novel_hash: str, vol_num: int, summary: Dict[str, Any], already: bool):
        """3-pass：壓縮 protagonist.background。已執行過就略過。"""
        if already:
            return summary, True
        before = (summary.get("protagonist") or {}).get("background") or ""
        if not before.strip():
            return summary, True
        self.console.print(f"  [cyan]3-pass 壓縮 background ({len(before)} chars)...[/cyan]")
        try:
            thought, new_bg, prompt = self.agent.compact_background(summary)
        except Exception as e:
            self.console.print(f"[red]Vol {vol_num} compact_background failed: {e}[/red]")
            return summary, False
        if new_bg and new_bg != before:
            new_summary = dict(summary)
            new_protag = dict(new_summary.get("protagonist") or {})
            new_protag["background"] = new_bg
            new_summary["protagonist"] = new_protag
            self._save_log(novel_hash, vol_num, 9999, {
                "stage": "compact_background",
                "prompt": prompt,
                "thought": thought,
                "background_before": before,
                "background_after": new_bg,
            })
            self.console.print(f"  [green]background 壓縮 {len(before)} → {len(new_bg)} chars[/green]")
            return new_summary, True
        return summary, True
