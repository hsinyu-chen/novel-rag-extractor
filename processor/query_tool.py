import json
from typing import List, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from processor.weaviate_storage import WeaviateStorage


class SearchWorldKnowledgeArgs(BaseModel):
    query_text: str = Field(
        default="",
        description=(
            "語意檢索關鍵字，例如：'長得像精靈的女孩'、'男主拿過的聖劍'。"
            "若只想用分類過濾或列舉全部，請留空字串。"
        ),
    )
    novel_hash: str = Field(
        default="",
        description=(
            "可選：限定某部小說的 hash（見 system prompt 列出的 hash 值）。"
            "留空字串 → 跨所有小說檢索。"
        ),
    )
    filter_type: str = Field(
        default="",
        description="可選：限定條目類別，只能填 character / location / object / concept / 空字串",
    )
    filter_categories: List[str] = Field(
        default_factory=list,
        description="可選：標籤陣列做精準過濾，例如 ['劍']、['反派','暗殺組織']。採 100% 精準比對。",
    )
    sort_by: str = Field(
        default="relevance",
        description="排序：relevance（預設，依檢索分數）或 appearances（依出場場景數，找主角/主要設定）",
    )
    limit: int = Field(default=5, description="回傳筆數上限，建議 3~8")


class SearchScenesArgs(BaseModel):
    query_text: str = Field(
        default="",
        description="場景語意檢索，例如：'男主第一次遇到聖劍'。留空則列舉指定範圍的所有 scene。",
    )
    novel_hash: str = Field(
        default="",
        description="可選：限定某部小說；留空 → 跨所有小說檢索。",
    )
    vol_num: int = Field(
        default=0,
        description="限定卷數（1-based），0 表示跨卷搜尋所有卷數。",
    )
    limit: int = Field(default=5, description="回傳筆數上限")


def build_query_tools(weaviate_db: WeaviateStorage) -> list:
    """
    將 WeaviateStorage 包裝成兩個 LangChain tool。
    不再綁定特定 novel_hash；LLM 可透過參數選擇目標小說（留空 = 跨 DB）。
    """

    def _search_world_knowledge(
        query_text: str = "",
        novel_hash: str = "",
        filter_type: str = "",
        filter_categories: Optional[List[str]] = None,
        sort_by: str = "relevance",
        limit: int = 5,
    ) -> str:
        results = weaviate_db.universal_search(
            novel_hash=novel_hash or "",
            max_vol=0,
            query_text=query_text,
            filter_type=filter_type,
            filter_categories=filter_categories or [],
            sort_by=sort_by,
            limit=max(1, min(limit, 15)),
        )
        if not results:
            return json.dumps({"hits": 0, "items": []}, ensure_ascii=False)
        slim = []
        for r in results:
            slim.append({
                "keyword": r.get("keyword"),
                "type": r.get("type"),
                "novel_hash": r.get("novel_hash"),
                "aliases": r.get("aliases") or [],
                "categories": r.get("categories") or [],
                "description": (r.get("description") or "")[:500],
                "appeared_in_count": len(r.get("appeared_in") or []),
                "vol_num": r.get("vol_num"),
            })
        return json.dumps({"hits": len(slim), "items": slim}, ensure_ascii=False)

    def _search_scenes(
        query_text: str = "",
        novel_hash: str = "",
        vol_num: int = 0,
        limit: int = 5,
    ) -> str:
        results = weaviate_db.search_scenes(
            novel_hash=novel_hash or "",
            query_text=query_text,
            vol_num=vol_num,
            limit=max(1, min(limit, 15)),
        )
        return json.dumps({"hits": len(results), "items": results}, ensure_ascii=False)

    world_tool = StructuredTool.from_function(
        func=_search_world_knowledge,
        name="search_world_knowledge",
        description=(
            "查詢小說世界觀知識庫（人物、地點、物品、概念），可跨多部小說檢索。"
            "novel_hash 留空 → 跨 DB；帶 hash → 鎖定單一作品。"
            "query_text 留空 + filter_categories 或 sort_by=appearances 可做精準列舉。"
        ),
        args_schema=SearchWorldKnowledgeArgs,
    )
    scene_tool = StructuredTool.from_function(
        func=_search_scenes,
        name="search_scenes",
        description=(
            "查詢場景分片的劇情摘要（Layer 1 chunks），可跨多部小說。"
            "novel_hash 留空 → 跨 DB；vol_num=0 → 跨卷；query_text 留空則列舉範圍內的 scene 大綱。"
        ),
        args_schema=SearchScenesArgs,
    )
    return [world_tool, scene_tool]
