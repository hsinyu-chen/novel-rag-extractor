import os
import hashlib
import json
from typing import List, Dict, Any

from langchain_core.runnables import RunnableLambda, RunnableConfig
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

from processor.knowledge_agent import KnowledgeAgent
from processor.json_storage import JsonStorage
from processor.weaviate_storage import WeaviateStorage

class KnowledgeProcess:
    """
    小說知識萃取處理引擎
    """
    def __init__(self, storage: JsonStorage, weaviate_db: WeaviateStorage, agent: KnowledgeAgent, config: Any):
        self.storage = storage
        self.weaviate_db = weaviate_db
        self.agent = agent
        self.conf = config
        self.console = Console()

        # LCEL Chain：Layer 1 寫入 → 抽取 → 逐 entity merge+upsert → 回填 chunk.entity_refs → 寫 scene JSON
        self.chain = (
            RunnableLambda(self._write_chunk_step) |
            RunnableLambda(self._extract_step) |
            RunnableLambda(self._merge_step) |
            RunnableLambda(self._backfill_chunk_refs_step) |
            RunnableLambda(self._save_step)
        )
        self.console.print("[bold green] Knowledge Process initialized.[/bold green]")

    def get_path_hash(self, path: str) -> str:
        return hashlib.md5(os.path.abspath(path).encode('utf-8')).hexdigest()[:8]

    def _write_chunk_step(self, scene_data: dict, config: RunnableConfig) -> dict:
        """LCEL Step 0: Layer 1 寫入 — 把 scene 原文 + title 向量化存進 NovelChunk"""
        params = config.get("configurable", {})
        progress = params.get("progress")
        tasks_ui = params.get("tasks_ui")
        novel_hash = params.get("novel_hash")
        vol_num = params.get("vol_num")

        scene_idx = scene_data.get("scene_index", 0)
        title = scene_data.get("title", "")
        content = scene_data.get("content", "")
        token_count = scene_data.get("token_count", 0)

        if progress and tasks_ui:
            progress.update(tasks_ui["chunk_task"], visible=True, description="[blue]Writing Chunk to Layer 1...")

        chunk_uuid = None
        try:
            chunk_uuid = self.weaviate_db.upsert_chunk(
                novel_hash, vol_num, scene_idx, title, content, token_count
            )
        except Exception as e:
            self.console.print(f"[red]Failed to upsert chunk for scene {scene_idx}: {e}[/red]")

        if progress and tasks_ui:
            progress.update(tasks_ui["chunk_task"], completed=100)

        return {
            "scene_data": scene_data,
            "chunk_uuid": chunk_uuid,
        }

    def _extract_step(self, data: dict, config: RunnableConfig) -> dict:
        """LCEL Step 1: 條目提取"""
        params = config.get("configurable", {})
        progress = params.get("progress")
        tasks_ui = params.get("tasks_ui")

        scene_data = data["scene_data"]
        content = scene_data.get("content", "")
        if progress and tasks_ui:
            progress.update(tasks_ui["extract_task"], visible=True, description="[cyan]Extracting Entities from Scene...")

        existing_types = params.get("existing_types", [])
        thought, result, prompt = self.agent.extract_entities(content, existing_types)

        if progress and tasks_ui:
            progress.update(tasks_ui["extract_task"], completed=100)

        # 記錄 Step 1 提取結果與 Prompt
        scene_idx = scene_data.get("scene_index", 0)
        novel_hash = params.get("novel_hash")
        self._save_log(novel_hash, f"scene_{scene_idx:03d}_extraction_step1_entities.json", {
            "scene_index": scene_idx,
            "prompt": prompt,
            "thought": thought,
            "result": result
        })

        return {
            **data,
            "extracted_entities": result.get("entities", [])
        }

    def _merge_step(self, data: dict, config: RunnableConfig) -> dict:
        """LCEL Step 2: Per-entity RAG 檢索 → LLM merge/create → inline upsert 到 Layer 2。

        Inline upsert 讓同 scene 後續別名能透過 RAG 看到剛寫入的前置條目。
        """
        params = config.get("configurable", {})
        progress = params.get("progress")
        tasks_ui = params.get("tasks_ui")
        novel_hash = params.get("novel_hash")

        vol_num = params.get("vol_num")

        scene_data = data["scene_data"]
        entities = data["extracted_entities"]
        scene_idx = scene_data.get("scene_index", 0)
        chunk_uuid = data.get("chunk_uuid")

        merged_entities = []
        saved_entity_uuids = []

        if progress and tasks_ui and entities:
            progress.update(tasks_ui["merge_task"], visible=True, total=len(entities), completed=0)

        # 取得場景原文，作為給 LLM 的輔助脈絡
        scene_content = scene_data.get("content", "")
        # 截取前半段 (2000 字) 足以判斷身分的長度即可
        scene_excerpt = scene_content[:2000] if scene_content else ""

        for i, entity in enumerate(entities):
            keyword = entity.get("keyword")
            e_type = entity.get("type", "world-setting")
            context = entity.get("context_summary", "")

            if not keyword:
                continue

            if progress and tasks_ui:
                progress.update(tasks_ui["merge_task"], description=f"[yellow]Merging Entity: {keyword} ({e_type})...")

            # Top-K RAG：受限於相同小說與卷數的候選人。
            # 雙軌門檻由 config 提供（rag_identity_strong/keep, rag_content_strong/min），調整走 .env。
            candidates = self.weaviate_db.search_similar_entity(
                novel_hash, vol_num, e_type, keyword, context, top_k=5
            )

            if not candidates:
                # [全然新建]：完全沒找到候選人，直接初始化條目
                thought, merged_result, prompt = self.agent.create_initial_entity(keyword, e_type, context, scene_idx, scene_excerpt=scene_excerpt)
            else:
                # [合併決策]：進入 LLM 判斷與整合流程
                llm_candidates = []
                cand_uuids = []
                for cand in candidates:
                    cand_copy = dict(cand)
                    cand_uuids.append(cand_copy.pop("_weaviate_uuid", None))
                    # 移除 LLM 不需要看到的 RAG 評分與偵錯欄位
                    cand_copy.pop("_score", None)
                    cand_copy.pop("_identity_sim", None)
                    cand_copy.pop("_content_sim", None)
                    cand_copy.pop("_match_track", None)
                    cand_copy.pop("appeared_in", None)
                    llm_candidates.append(cand_copy)

                thought, merged_result, prompt = self.agent.merge_entity(keyword, e_type, context, llm_candidates, scene_idx, scene_excerpt=scene_excerpt)

            if merged_result:
                selected_idx = merged_result.get("selected_index", -1)
                existing_uuid = None
                merged_data = dict(merged_result)
                
                # 1. 核心繼承邏輯
                if selected_idx >= 0 and selected_idx < len(cand_uuids):
                    existing_uuid = cand_uuids[selected_idx]
                    old_obj = candidates[selected_idx]
                    
                    # [場景繼承]
                    merged_data["appeared_in"] = old_obj.get("appeared_in", [])

                    # [Chunk 繼承]：保留舊實體累積的 chunk_refs，新 chunk_uuid 會在 upsert_entity 內做 union
                    merged_data["chunk_refs"] = list(old_obj.get("chunk_refs", []) or [])

                    # [名稱權限與別名優先級]
                    # 如果舊名稱不是「未知」，則強制鎖定舊名稱為 Canonical Name
                    old_keyword = old_obj.get("keyword", "")
                    if "未知" not in old_keyword and old_keyword != keyword:
                        merged_data["keyword"] = old_keyword
                        # 將新提取的名稱存入別名
                        if keyword not in merged_data["aliases"]:
                            merged_data["aliases"].append(keyword)

                    # [別名整合]：聯集舊實體的別名
                    existing_aliases = old_obj.get("aliases", [])
                    merged_data["aliases"] = list(set((merged_data.get("aliases") or []) + existing_aliases))

                    # [狀態變更整合]：聯集舊實體的狀態記錄
                    existing_status = old_obj.get("major_status_changes", [])
                    new_status = merged_data.get("major_status_changes", [])
                    # 以 scene_index + event 作為唯一性判斷 (簡單過濾)
                    combined_status = existing_status + [ns for ns in new_status if ns not in existing_status]
                    merged_data["major_status_changes"] = sorted(combined_status, key=lambda x: x.get("scene_index", 0))

                # 2. 清理與標準化
                merged_data.pop("selected_index", None)
                merged_data["type"] = e_type 

                # 3. 寫入偵錯 Log
                log_prefix = "merge" if existing_uuid else "new"
                log_filename = f"scene_{scene_idx:03d}_extraction_{log_prefix}_{keyword}_{hashlib.md5(str(existing_uuid or '').encode()).hexdigest()[:8]}.json"
                self._save_log(novel_hash, log_filename, {
                    "scene_index": scene_idx,
                    "keyword": keyword,
                    "type": e_type,
                    "candidates_count": len(candidates),
                    "rag_candidates": candidates,
                    "full_prompt": prompt,
                    "thought": thought,
                    "result": merged_data,
                    "existing_uuid": existing_uuid
                })

                # 最後防線：如果經過 LLM 處理後 keyword 或 description 仍然無效，則拒絕存入
                final_kw = str(merged_data.get("keyword", "")).strip().upper()
                final_desc = str(merged_data.get("description", "")).strip().upper()
                invalid_tokens = {"N/A", "NONE", "NULL", "", "無", "未知", "無資料", "無相關資訊"}
                if final_kw in invalid_tokens:
                    self.console.print(f"[yellow]Warning: Skipped entity with invalid keyword: '{keyword}'[/yellow]")
                elif final_desc in invalid_tokens:
                    self.console.print(f"[yellow]Warning: Skipped entity '{keyword}' due to empty/N/A description[/yellow]")
                else:
                    # Inline upsert：讓同 scene 後續條目可透過 RAG 看到此條目
                    e_type_final = merged_data.get("type", e_type)
                    try:
                        uuid_key = self.weaviate_db.upsert_entity(
                            novel_hash, vol_num, merged_data, existing_uuid, scene_idx, chunk_uuid=chunk_uuid
                        )
                    except Exception as ex:
                        self.console.print(f"[red]Failed to upsert entity '{keyword}': {ex}[/red]")
                        if progress and tasks_ui:
                            progress.update(tasks_ui["merge_task"], advance=1)
                        continue

                    # 動態擴充現存的類型庫，後續 scene 的 extract_entities 能看到它
                    if e_type_final not in params["existing_types"]:
                        params["existing_types"].append(e_type_final)

                    saved_entity_uuids.append(uuid_key)
                    merged_entities.append({
                        "keyword": merged_data["keyword"],
                        "merged_data": merged_data,
                        "uuid": uuid_key,
                        "type": e_type_final,
                    })

            if progress and tasks_ui:
                progress.update(tasks_ui["merge_task"], advance=1)

        return {
            **data,
            "merged_entities": merged_entities,
            "saved_entity_uuids": saved_entity_uuids,
        }

    def _backfill_chunk_refs_step(self, data: dict, config: RunnableConfig) -> dict:
        """LCEL Step 3: 回填 chunk.entity_refs — 讓 Layer 1 也能反查本 scene 抽取到的所有 entity"""
        chunk_uuid = data.get("chunk_uuid")
        entity_uuids = data.get("saved_entity_uuids", [])
        if chunk_uuid and entity_uuids:
            try:
                self.weaviate_db.update_chunk_entity_refs(chunk_uuid, entity_uuids)
            except Exception as e:
                self.console.print(f"[red]Failed to backfill chunk_refs for {chunk_uuid}: {e}[/red]")
        return data

    def _save_step(self, data: dict, config: RunnableConfig) -> dict:
        """LCEL Step 4: 寫本地 JSON 備份 (world/*.json + scene_XXX.json)。

        本地 JSON 為人類可讀的 debug / 還原用備份，Weaviate 寫入於 _merge_step 完成。
        """
        params = config.get("configurable", {})
        progress = params.get("progress")
        tasks_ui = params.get("tasks_ui")
        scene_key = params.get("scene_key")

        novel_hash = params.get("novel_hash")
        vol_num = params.get("vol_num")

        scene_data = data["scene_data"]
        merged_entities = data["merged_entities"]
        chunk_uuid = data.get("chunk_uuid")

        if progress and tasks_ui:
            progress.update(tasks_ui["save_task"], visible=True, description="[magenta]Saving updates...")

        entities_extracted = []
        for entity_info in merged_entities:
            uuid_key = entity_info["uuid"]
            e_type = entity_info["type"]

            # 本機 JSON 備份（以 UUID 為檔名）
            local_key = f"{novel_hash}.world.vol_{vol_num}.{e_type}.{uuid_key}"
            self.storage.set(local_key, entity_info["merged_data"])

            entities_extracted.append({
                "keyword": entity_info["keyword"],
                "type": e_type,
                "uuid": uuid_key,
                "chunk_uuid": chunk_uuid,
            })

        # 擴充 scene_data：新增 chunk_uuid 欄位 + entities_extracted 關聯
        scene_data["chunk_uuid"] = chunk_uuid
        scene_data["entities_extracted"] = entities_extracted
        self.storage.set(scene_key, scene_data)

        if progress and tasks_ui:
            progress.update(tasks_ui["save_task"], completed=100)

        return data

    def _save_log(self, novel_hash: str, filename: str, log_data: Any):
        """
        儲存萃取與合併過程的偵錯 Log
        路徑: output/<hash>/logs/extraction/<filename>.json
        """
        log_dir = os.path.join("output", novel_hash, "logs", "extraction")
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, filename)
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[KnowledgeProcess] Failed to save log {filename}: {e}")

    def run(self, novel_name: str, start_vol: int, end_vol: int = 0, clean_output: bool = False):
        base_data_path = os.path.join("data", novel_name)
        if not os.path.exists(base_data_path):
            self.console.print(f"[red]Error: 找不到小說資料夾 {base_data_path}[/red]")
            return
        
        novel_hash = self.get_path_hash(base_data_path)

        # --clean：清空本卷 world 與 extraction logs 目錄
        if clean_output:
            import shutil
            world_dir = os.path.join("output", novel_hash, "world")
            if os.path.exists(world_dir):
                shutil.rmtree(world_dir)
                self.console.print(f"[yellow]已清除 World Knowledge 目錄: {world_dir}[/yellow]")
            
            log_dir = os.path.join("output", novel_hash, "logs", "extraction")
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
                self.console.print(f"[yellow]已清除 Extraction Logs 目錄: {log_dir}[/yellow]")

        self.console.print(f"\n[bold magenta]>> 開始知識提取: {novel_name}[/bold magenta] (Hash: {novel_hash})")

        # 預先撈取現存的 Entity Types，做為 Zero-Shot 動態分類的參考錨點
        shared_existing_types = self.weaviate_db.get_existing_entity_types()
        self.console.print(f"[green]>> 已初始化條目類型庫: {', '.join(shared_existing_types)}[/green]")

        vol_keys = []
        vol_dirs = self.storage.list_namespaces(f"{novel_hash}.scenes.processed")

        # 過濾出需要處理的卷數（形式為 "<hash>.scenes.processed.vol_N"）
        for vk in vol_dirs:
            vol_str = vk.split('.')[-1]
            if not vol_str.startswith("vol_"): continue
            v_num = int(vol_str.split('_')[-1])
            if start_vol <= v_num and (end_vol == 0 or v_num <= end_vol):
                vol_keys.append((v_num, vk))

        vol_keys.sort()

        for v_num, vk in vol_keys:
            self.console.print(f"\n[cyan]>> Processing Volume {v_num}...[/cyan]")
            
            if clean_output:
                self.console.print(f"[yellow]>> Clearing Weaviate Data for Volume {v_num}...[/yellow]")
                self.weaviate_db.clear_novel_volume(novel_hash, v_num)
            
            # 撈出該集下的所有 scenes 並依檔名排序
            scene_keys = self.storage.list_keys(vk)
            scene_keys.sort()

            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                BarColumn(), TaskProgressColumn(), console=self.console,
            ) as progress:
                overall_task = progress.add_task(f"[bold blue]Volume {v_num} scenes", total=len(scene_keys))

                for scene_key in scene_keys:
                    scene_data = self.storage.get(scene_key)
                    if not scene_data or not scene_data.get("content"):
                        progress.update(overall_task, advance=1)
                        continue

                    # UI tasks
                    tasks_ui = {
                        "chunk_task": progress.add_task("[blue]Chunking...", total=100, visible=False),
                        "extract_task": progress.add_task("[cyan]Extracting...", total=100, visible=False),
                        "merge_task": progress.add_task("[yellow]Merging...", total=100, visible=False),
                        "save_task": progress.add_task("[magenta]Saving...", total=100, visible=False)
                    }

                    # Invoke LCEL
                    try:
                        self.chain.invoke(scene_data, config={
                            "configurable": {
                                "progress": progress,
                                "tasks_ui": tasks_ui,
                                "novel_hash": novel_hash,
                                "vol_num": v_num,
                                "scene_key": scene_key,
                                "existing_types": shared_existing_types
                            }
                        })
                    except Exception as e:
                        self.console.print(f"[red]Error on {scene_key}: {e}[/red]")

                    # Cleanup tasks
                    for tid in tasks_ui.values():
                        progress.remove_task(tid)
                        
                    progress.update(overall_task, advance=1)
        
        self.console.print(f"\n[bold green]{novel_name} 知識分析與萃取完成。[/bold green]")
