import os
import re
import hashlib
import json
from typing import List, Dict, Any
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from langchain_core.runnables import RunnableLambda, RunnableConfig
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

from core.config import PipelineConfig
from processor.llm_engine import NativeLlamaEngine
from processor.scene_validator import SceneValidator
from processor.scene_summarizer import SceneSummarizer

# ── 結構性分隔符 regex（語義分詞前的硬切） ──
# 匹配順序：章節標題 > 裝飾分隔符 > 線段分隔符 > 連續空行
_STRUCTURAL_BREAK = re.compile(
    r"(?:"
    # 1. 中文章節標題：第X卷、第X章、第X節、第X話、第X回（前後有換行）
    r"(?=\n\s*第\s*?[一二三四五六七八九十百千\d]+\s*?[卷章節話回篇]\b)"
    r"|"
    # 2. 英文章節標題：Chapter X, Prologue, Epilogue
    r"(?=\n\s*(?:chapter|prologue|epilogue)\s)"
    r"|"
    # 3. 裝飾分隔符：◆ ◇ ★ ☆ ■ □ ● ○ ※ ＊ *（含全形）可重複
    r"(?=\n\s*[◆◇★☆■□●○※＊\*]{1,}\s*\n)"
    r"|"
    # 4. 線段分隔符：--- === ─── ━━━ ＝＝＝（3個以上）
    r"(?=\n\s*[-=─━＝]{3,}\s*\n)"
    r"|"
    # 5. 連續 3+ 空行
    r"(?=\n{3,})"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


class BookPreProcessor:
    """
    小說預處理引擎
    """
    def __init__(self, storage, validator: SceneValidator, summarizer: SceneSummarizer, config: Any):
        self.storage = storage
        self.validator = validator
        self.summarizer = summarizer
        self.conf = config
        self.n_ctx = int(self.conf["n_ctx"])
        self.n_gpu_layers = int(self.conf["n_gpu_layers"])
        self.console = Console()

        # 1. 初始化模型
        self._init_models()

        # 2. 宣告式 LCEL Chain 組裝
        # 我們將切分與驗證包裝成獨立的 RunnableStep
        self.chain = (
            RunnableLambda(self._split_step) | 
            RunnableLambda(self._validate_step) |
            RunnableLambda(self._summarize_and_save_step)
        )
        
        self.console.print("[bold green] Pre-processor initialized.[/bold green]")

    def _init_models(self):
        # Embedding (TODO: Could also be moved to DI Container)
        embed_path = hf_hub_download(repo_id=self.conf["embed_model_repo"], filename=self.conf["embed_model_file"])
        self.embed_engine = Llama(model_path=embed_path, n_gpu_layers=self.n_gpu_layers, n_ctx=int(self.conf["embed_n_ctx"]), embedding=True, verbose=False)
        
        class LlamaSimpleEmbeddings:
            def __init__(self, engine): self.engine = engine
            def embed_documents(self, t): return [self.engine.create_embedding(i)["data"][0]["embedding"] for i in t]
            def embed_query(self, t): return self.engine.create_embedding(t)["data"][0]["embedding"]
        
        from langchain_experimental.text_splitter import SemanticChunker
        self.chunker = SemanticChunker(
            LlamaSimpleEmbeddings(self.embed_engine),
            sentence_split_regex=r"(?<=[。！？\n\n])",
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=99,
            buffer_size=3
        )

    @staticmethod
    def _structural_presplit(text: str) -> List[str]:
        """在語義分詞之前，先用 regex 硬切章節/分隔符邊界。"""
        sections = _STRUCTURAL_BREAK.split(text)
        return [s for s in sections if s.strip()]

    def _split_step(self, text: str, config: RunnableConfig) -> List[str]:
        """LCEL Step 1: 結構硬切 → 語義切分"""
        params = config.get("configurable", {})
        progress = params.get("progress")
        file_task = params.get("file_task")
        filename = params.get("filename")

        if progress and file_task:
            progress.update(file_task, description=f"[cyan]Structural Pre-split {filename}...", completed=10)

        # Phase 1: 結構性硬切（章節、分隔符、連續空行）
        sections = self._structural_presplit(text)

        if progress and file_task:
            progress.update(file_task, description=f"[cyan]Semantic Splitting {filename} ({len(sections)} sections)...", completed=20)

        # Phase 2: 每個 section 內部做語義分詞
        all_chunks: List[str] = []
        for section in sections:
            chunks = [s for s in self.chunker.split_text(section) if s.strip()]
            all_chunks.extend(chunks)

        if progress and file_task:
            progress.update(file_task, completed=40)

        return all_chunks

    def _validate_step(self, initial_scenes: List[str], config: RunnableConfig) -> List[str]:
        """LCEL Step 2: 場景邊界驗證"""
        params = config.get("configurable", {})
        progress = params.get("progress")
        file_task = params.get("file_task")
        scene_task = params.get("scene_task")
        filename = params.get("filename")
        novel_name = params.get("novel_name")
        vol_num = params.get("vol_num")
        novel_hash = params.get("novel_hash")
        line_count = params.get("line_count")

        if not initial_scenes:
            return []

        if progress and scene_task:
            progress.update(scene_task, visible=True, total=len(initial_scenes)-1, completed=0)

        def on_boundary(idx, total, count, snippet):
            if progress:
                progress.update(scene_task, completed=idx, description=f"[yellow]Gemma-4 Validating: {idx}/{total} - Found {count} scenes")
                progress.update(file_task, description=f"[cyan]Processing {filename} ({line_count} lines) -> {count} scenes")

        def on_pre_save(scene_idx, content):
            # 中間狀態存檔 (防止 Step 3 崩潰導致整集白做)
            token_count = self.count_tokens(content)
            key = f"{novel_hash}.scenes.processed.vol_{vol_num}.scene_{scene_idx:03d}"
            self.storage.set(key, {
                "novel": novel_name, "volume": vol_num, "scene_index": scene_idx,
                "token_count": token_count, "title": f"Scene {scene_idx} (Segmented)",
                "content": content, "status": "segmented"
            })

        # 執行驗證 (在此階段進行中間存檔)
        log_dir = os.path.join("output", novel_hash, "logs", "validate", f"vol_{vol_num}")
        final_scenes = self.validator.validate_boundaries(
            initial_scenes, 
            on_boundary_checked=on_boundary, 
            on_scene_ready=on_pre_save,
            log_dir=log_dir,
            max_tokens=4096,
            tokenizer=self.count_tokens,
        )
        
        return final_scenes

    def _summarize_and_save_step(self, final_scenes: List[str], config: RunnableConfig) -> List[str]:
        """LCEL Step 3: 場景摘要與存檔"""
        params = config.get("configurable", {})
        progress = params.get("progress")
        file_task = params.get("file_task")
        scene_task = params.get("scene_task")
        novel_name = params.get("novel_name")
        vol_num = params.get("vol_num")
        novel_hash = params.get("novel_hash")

        if not final_scenes:
            if progress and file_task:
                progress.update(file_task, completed=100)
            return []

        if progress and scene_task:
            progress.update(scene_task, visible=True, total=len(final_scenes), completed=0, description="[bold magenta]Gemma-4 Summarizing...")

        last_summary = ""
        for i, content in enumerate(final_scenes):
            scene_idx = i + 1
            if progress and scene_task:
                progress.update(scene_task, completed=i, description=f"[bold magenta]Gemma-4 Summarizing: {scene_idx}/{len(final_scenes)}")

            # 1. 產出摘要
            thought, summary = self.summarizer.summarize_scene(content, last_summary)
            last_summary = summary

            # 2. 計算 Token
            token_count = self.count_tokens(content)
            if token_count > self.n_ctx:
                raise ValueError(f"Scene {scene_idx} too long: {token_count} tokens")
            
            # 3. 存檔
            key = f"{novel_hash}.scenes.processed.vol_{vol_num}.scene_{scene_idx:03d}"
            self.storage.set(key, {
                "novel": novel_name, "volume": vol_num, "scene_index": scene_idx,
                "token_count": token_count, "title": summary,
                "content": content, "status": "segmented"
            })

        if progress:
            if file_task: progress.update(file_task, completed=100)
            if scene_task: progress.update(scene_task, visible=False)
            
        return final_scenes

    def count_tokens(self, text: str) -> int:
        return len(self.embed_engine.tokenize(text.encode("utf-8"), add_bos=False))

    def get_path_hash(self, path: str) -> str:
        return hashlib.md5(os.path.abspath(path).encode('utf-8')).hexdigest()[:8]

    def get_numeric_value(self, filename: str) -> int:
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0

    def run(self, novel_name: str, start_vol: int, end_vol: int = 0, clean_output: bool = False):
        base_data_path = os.path.join("data", novel_name)
        if not os.path.exists(base_data_path):
            self.console.print(f"[red]Error: 找不到小說資料夾 {base_data_path}[/red]")
            return
        
        novel_hash = self.get_path_hash(base_data_path)

        # --clean: 清除該小說的全部輸出
        if clean_output:
            import shutil
            output_dir = os.path.join("output", novel_hash)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                self.console.print(f"[yellow]已清除輸出目錄: {output_dir}[/yellow]")

        self.console.print(f"\n[bold magenta]>> 開始預處理: {novel_name}[/bold magenta] (Hash: {novel_hash})")

        all_files = [f for f in os.listdir(base_data_path) if f.endswith(".txt")]
        target_files = sorted([(self.get_numeric_value(f), f) for f in all_files if self.get_numeric_value(f) >= start_vol])
        # --vol: 只跑指定的單一集數
        if end_vol > 0:
            target_files = [(n, f) for n, f in target_files if n <= end_vol]

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=self.console,
        ) as progress:
            overall_task = progress.add_task("[bold blue]Overall Progress", total=len(target_files))
            
            for vol_num, filename in target_files:
                file_path = os.path.join(base_data_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    full_text = "".join(lines)
                
                # 建立當前檔案的任務
                file_task = progress.add_task(f"[cyan]Ingesting {filename}...", total=100)
                scene_task = progress.add_task("[yellow]Validating...", total=100, visible=False)

                # 使用 LCEL Chain 執行
                self.chain.invoke(full_text, config={
                    "configurable": {
                        "progress": progress,
                        "file_task": file_task,
                        "scene_task": scene_task,
                        "filename": filename,
                        "novel_name": novel_name,
                        "vol_num": vol_num,
                        "novel_hash": novel_hash,
                        "line_count": len(lines)
                    }
                })
                
                progress.remove_task(file_task)
                progress.remove_task(scene_task)
                progress.update(overall_task, advance=1)
        
        self.console.print(f"\n[bold green]{novel_name} 預處理完成。[/bold green]")
