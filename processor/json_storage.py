import os
import json
from typing import Any, Dict, List, Optional
from core.storage import BaseStorage

class JsonStorage(BaseStorage):
    def __init__(self, base_dir: str = "output"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _key_to_path(self, key: str) -> str:
        """
        將 'a.b.c' 轉換為 'base_dir/a/b/c.json'
        """
        parts = key.split('.')
        # 除了最後一個元素外，其餘都是資料夾
        dir_path = os.path.join(self.base_dir, *parts[:-1])
        file_name = f"{parts[-1]}.json"
        
        # 自動建立目錄
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, file_name)

    def set(self, key: str, value: Any):
        """
        實作 KV 儲存，自動處理目錄結構
        """
        file_path = self._key_to_path(key)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=4)
            # print(f"Stored: {key} -> {file_path}")
        except Exception as e:
            print(f"Error storing key {key}: {e}")

    def get(self, key: str) -> Optional[Any]:
        """
        讀取 JSON 資料
        """
        file_path = self._key_to_path(key)
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading key {key}: {e}")
            return None

    def list_keys(self, prefix: str) -> List[str]:
        """
        列出特定目錄下的所有子 Key
        """
        parts = prefix.split('.')
        target_dir = os.path.join(self.base_dir, *parts)
        
        if not os.path.exists(target_dir):
            return []
            
        keys = []
        for f in os.listdir(target_dir):
            if f.endswith(".json"):
                keys.append(f"{prefix}.{f[:-5]}")
        return sorted(keys)

    def get_history_summary(self, limit: int = 5) -> str:
        """
        獲取最近場景的簡短語義回顧 (掃描 scenes 目錄)
        """
        # 這裡示範如何利用 list_keys 來實作
        summaries = []
        # 假設結構是 scenes.chapter_xxx
        all_scene_keys = self.list_keys("scenes")
        
        for key in all_scene_keys[-limit:]:
            data = self.get(key)
            if data:
                summaries.append(f"[{key}] {data.get('summary', 'N/A')}")
        
        return "\n".join(summaries)
