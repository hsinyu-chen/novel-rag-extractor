from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseStorage(ABC):
    """
    負責中間產物儲存的基礎類別
    支援命名空間路徑，使用 '.' 連接
    """

    @abstractmethod
    def set(self, key: str, value: Any):
        """
        將資料存入指定的路徑
        key 範例: 'scenes.chapter1.scene1'
        """
        pass

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        根據路徑讀取資料
        """
        pass

    @abstractmethod
    def list_keys(self, prefix: str) -> List[str]:
        """
        列出特定前綴下的所有 Key (例如 'characters.main')
        """
        pass

    # --- 快捷方法 (Legacy/Convention Support) ---
    
    def save_scene(self, chapter_id: str, scene_id: str, data: Dict[str, Any]):
        self.set(f"scenes.{chapter_id}.{scene_id}", data)

    def update_state(self, category: str, item_name: str, data: Dict[str, Any]):
        self.set(f"states.{category}.{item_name}", data)

    @abstractmethod
    def get_history_summary(self, limit: int = 5) -> str:
        """獲取最近的摘要，供 Agent 參考連貫性"""
        pass
