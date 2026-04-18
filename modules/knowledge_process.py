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

        # 宣告式 LCEL Chain 組裝
        self.chain = (
            RunnableLambda(self._extract_step) |
            RunnableLambda(self._merge_step) |
            RunnableLambda(self._save_step)
        )
        self.console.print("[bold green] Knowledge Process initialized.[/bold green]")

    def get_path_hash(self, path: str) -> str:
        return hashlib.md5(os.path.abspath(path).encode('utf-8')).hexdigest()[:8]

    def _extract_step(self, scene_data: dict, config: RunnableConfig) -> dict:
        """LCEL Step 1: 條目提取"""
        params = config.get("configurable", {})
        progress = params.get("progress")
        tasks_ui = params.get("tasks_ui")
        
        content = scene_data.get("content", "")
        if progress and tasks_ui:
            progress.update(tasks_ui["extract_task"], visible=True, description="[cyan]Extracting Entities from Scene...")

        existing_types = params.get("existing_types", [])
        thought, result = self.agent.extract_entities(content, existing_types)
        
        if progress and tasks_ui:
            progress.update(tasks_ui["extract_task"], completed=100)

        # 傳遞萃取到的條目到下一個步驟，同時也保留原始資料
        return {
            "scene_data": scene_data,
            "extracted_entities": result.get("entities", [])
        }

    def _merge_step(self, data: dict, config: RunnableConfig) -> dict:
        """LCEL Step 2: 檔案檢閱與合併"""
        params = config.get("configurable", {})
        progress = params.get("progress")
        tasks_ui = params.get("tasks_ui")
        novel_hash = params.get("novel_hash")

        vol_num = params.get("vol_num")

        scene_data = data["scene_data"]
        entities = data["extracted_entities"]
        scene_idx = scene_data.get("scene_index", 0)

        merged_entities = []

        if progress and tasks_ui and entities:
            progress.update(tasks_ui["merge_task"], visible=True, total=len(entities), completed=0)

        for i, entity in enumerate(entities):
            keyword = entity.get("keyword")
            e_type = entity.get("type", "world-setting")
            context = entity.get("context_summary", "")

            if not keyword:
                continue

            if progress and tasks_ui:
                progress.update(tasks_ui["merge_task"], description=f"[yellow]Merging Entity: {keyword} ({e_type})...")

            # 從 Weaviate 進行語義搜索，拿回最符合的 UUID 與 JSON 結構 (限制搜索上限為目前的 vol_num)
            existing_data = self.weaviate_db.search_similar_entity(novel_hash, vol_num, e_type, keyword, context)
            existing_uuid = existing_data.pop("_weaviate_uuid", None) if existing_data else None

            if e_type == "character":
                _, merged_data = self.agent.merge_character(keyword, context, existing_data, scene_idx)
            else:
                _, merged_data = self.agent.merge_generic_entity(keyword, e_type, context, existing_data)

            if merged_data:
                merged_entities.append({
                    "keyword": keyword,
                    "merged_data": merged_data,
                    "existing_uuid": existing_uuid
                })

            if progress and tasks_ui:
                progress.update(tasks_ui["merge_task"], advance=1)

        return {
            "scene_data": scene_data,
            "merged_entities": merged_entities
        }

    def _save_step(self, data: dict, config: RunnableConfig) -> dict:
        """LCEL Step 3: 更新儲存庫與標註原始切片"""
        params = config.get("configurable", {})
        progress = params.get("progress")
        tasks_ui = params.get("tasks_ui")
        scene_key = params.get("scene_key")

        novel_hash = params.get("novel_hash")
        vol_num = params.get("vol_num")

        scene_data = data["scene_data"]
        merged_entities = data["merged_entities"]
        scene_idx = scene_data.get("scene_index", 0)

        if progress and tasks_ui:
            progress.update(tasks_ui["save_task"], visible=True, description="[magenta]Saving updates...")

        entities_extracted = []
        for entity_info in merged_entities:
            # 1. 存入 Weaviate VectorDB (UPSERT)
            e_type = entity_info["merged_data"].get("type", "unknown")
            uuid_key = self.weaviate_db.upsert_entity(novel_hash, vol_num, entity_info["merged_data"], entity_info["existing_uuid"], scene_idx)
            
            # 動態擴充現存的類型庫
            if e_type not in params["existing_types"]:
                params["existing_types"].append(e_type)
            
            # 2. 存入本機 JsonStorage 備份 (檔名使用 UUID)
            local_key = f"{novel_hash}.world.vol_{vol_num}.{e_type}.{uuid_key}"
            self.storage.set(local_key, entity_info["merged_data"])
            
            entities_extracted.append({
                "keyword": entity_info["keyword"],
                "type": e_type,
                "uuid": uuid_key
            })

        # 3. 幫原本的 scene_data 加上豐富綁定條目資訊
        # 防呆，假設之前有用 keywords 字串陣列，直接改版覆寫為關聯物件陣列
        scene_data["entities_extracted"] = entities_extracted

        # 4. 把 scene 存回原本的位置
        self.storage.set(scene_key, scene_data)

        if progress and tasks_ui:
            progress.update(tasks_ui["save_task"], completed=100)

        return data

    def run(self, novel_name: str, start_vol: int, end_vol: int = 0, clean_output: bool = False):
        base_data_path = os.path.join("data", novel_name)
        if not os.path.exists(base_data_path):
            self.console.print(f"[red]Error: 找不到小說資料夾 {base_data_path}[/red]")
            return
        
        novel_hash = self.get_path_hash(base_data_path)

        # --clean 機制: 這裡如果是 clean，是否清空 world 庫？
        if clean_output:
            import shutil
            world_dir = os.path.join("output", novel_hash, "world")
            if os.path.exists(world_dir):
                shutil.rmtree(world_dir)
                self.console.print(f"[yellow]已清除 World Knowledge 目錄: {world_dir}[/yellow]")

        self.console.print(f"\n[bold magenta]>> 開始知識提取: {novel_name}[/bold magenta] (Hash: {novel_hash})")

        # 預先撈取現存的 Entity Types，做為 Zero-Shot 動態分類的參考錨點
        shared_existing_types = self.weaviate_db.get_existing_entity_types()
        self.console.print(f"[green]>> 已初始化條目類型庫: {', '.join(shared_existing_types)}[/green]")

        vol_keys = []
        vol_dirs = self.storage.list_namespaces(f"{novel_hash}.scenes.processed")
        
        # list_keys 的回傳形式大概是 "<hash>.scenes.processed.vol_1", 所以我們要濾出要跑的集數
        for vk in vol_dirs:
            vol_str = vk.split('.')[-1] # "vol_1"
            if not vol_str.startswith("vol_"): continue
            v_num = int(vol_str.split('_')[-1])
            if start_vol <= v_num and (end_vol == 0 or v_num <= end_vol):
                vol_keys.append((v_num, vk))
        
        vol_keys.sort() # 按卷數小到大

        for v_num, vk in vol_keys:
            self.console.print(f"\n[cyan]>> Processing Volume {v_num}...[/cyan]")
            
            if clean_output:
                self.console.print(f"[yellow]>> Clearing Weaviate Data for Volume {v_num}...[/yellow]")
                self.weaviate_db.clear_novel_volume(novel_hash, v_num)
            
            # 撈出該集下的所有 scenes
            scene_keys = self.storage.list_keys(vk)
            # 因為 list_keys 是找第一層子檔，如果是資料夾會被略過，但 scenes 都是檔名，所以沒問題
            # 確認撈出的順序是對的
            scene_keys.sort()

            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                BarColumn(), TaskProgressColumn(), console=self.console,
            ) as progress:
                overall_task = progress.add_task(f"[bold blue]Volume {v_num} scenes", total=len(scene_keys))

                for scene_key in scene_keys:
                    # e.g scene_key = novel_hash.scenes.processed.vol_1.scene_001
                    scene_data = self.storage.get(scene_key)
                    if not scene_data or not scene_data.get("content"):
                        progress.update(overall_task, advance=1)
                        continue

                    # UI tasks
                    tasks_ui = {
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
