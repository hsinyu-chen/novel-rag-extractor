import yaml
from typing import Any, List, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from processor.weaviate_storage import WeaviateStorage
from processor.json_storage import JsonStorage


def _dump(obj: Any) -> str:
    """統一 tool 回傳格式：YAML block style，保留中文，關閉 key 排序。"""
    return yaml.safe_dump(
        obj,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=1000,
    )


def _strip_internal(d: dict) -> dict:
    """剔除給 LLM 無意義的欄位：uuid、內部 RAG 分數（開頭 _ 的 key）。"""
    if not isinstance(d, dict):
        return d
    return {
        k: v for k, v in d.items()
        if not k.startswith("_") and k not in ("uuid",)
    }


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


class GetSceneContentArgs(BaseModel):
    novel_hash: str = Field(
        description="必填：小說 hash（見 system prompt 的【資料庫收錄】區塊）。",
    )
    vol_num: int = Field(
        description="必填：卷數（1-based）。",
    )
    scene_index: int = Field(
        description="必填：場景索引（由 search_scenes 回傳的 scene_index 欄位得知）。",
    )


class GetVolSummaryArgs(BaseModel):
    novel_hash: str = Field(
        description="必填：小說 hash（見 system prompt 的【資料庫收錄】區塊）。",
    )
    vol_num: int = Field(
        description="必填：卷數（1-based）。",
    )


def build_query_tools(weaviate_db: WeaviateStorage, storage: Optional[JsonStorage] = None) -> list:
    """
    將 WeaviateStorage / JsonStorage 包裝成 LangChain tools。
    LLM 可透過 novel_hash 參數選擇目標小說（留空 = 跨 DB）。
    回傳格式統一為 YAML（block style），自動剔除 uuid 與內部 RAG 分數欄位。
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
            return _dump({"hits": 0, "items": []})
        slim = []
        for r in results:
            slim.append({
                "keyword": r.get("keyword"),
                "type": r.get("type"),
                "novel_hash": r.get("novel_hash"),
                "vol_num": r.get("vol_num"),
                "aliases": r.get("aliases") or [],
                "categories": r.get("categories") or [],
                "description": (r.get("description") or "")[:500],
                "appeared_in_count": len(r.get("appeared_in") or []),
            })
        return _dump({"hits": len(slim), "items": slim})

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
        items = [_strip_internal(r) for r in results]
        return _dump({"hits": len(items), "items": items})

    def _get_scene_content(
        novel_hash: str,
        vol_num: int,
        scene_index: int,
    ) -> str:
        data = weaviate_db.get_scene_content(
            novel_hash=novel_hash,
            vol_num=int(vol_num),
            scene_index=int(scene_index),
        )
        if not data:
            return _dump({
                "found": False,
                "novel_hash": novel_hash,
                "vol_num": vol_num,
                "scene_index": scene_index,
            })
        return _dump({"found": True, **_strip_internal(data)})

    world_tool = StructuredTool.from_function(
        func=_search_world_knowledge,
        name="search_world_knowledge",
        description=(
            "查詢小說世界觀知識庫（人物、地點、物品、概念），可跨多部小說檢索。"
            "novel_hash 留空 → 跨 DB；帶 hash → 鎖定單一作品。"
            "query_text 留空 + filter_categories 或 sort_by=appearances 可做精準列舉。"
            "回傳 YAML 格式。"
        ),
        args_schema=SearchWorldKnowledgeArgs,
    )
    scene_tool = StructuredTool.from_function(
        func=_search_scenes,
        name="search_scenes",
        description=(
            "查詢場景分片的劇情摘要（Layer 1 chunks），可跨多部小說。"
            "novel_hash 留空 → 跨 DB；vol_num=0 → 跨卷；query_text 留空則列舉範圍內的 scene 大綱。"
            "只回 title + 400 字摘要；需要原文對話、暱稱、細節請接著用 get_scene_content。"
            "回傳 YAML 格式。"
        ),
        args_schema=SearchScenesArgs,
    )
    scene_content_tool = StructuredTool.from_function(
        func=_get_scene_content,
        name="get_scene_content",
        description=(
            "按 (novel_hash, vol_num, scene_index) 取單一 scene 的完整原文；"
            "用於 search_scenes 摘要不足以回答時，drill-down 到對話 / 暱稱 / 具體細節。"
            "單次呼叫一個 scene；token_count 會一併回傳供 agent 自行控制預算。"
            "回傳 YAML 格式。"
        ),
        args_schema=GetSceneContentArgs,
    )

    def _get_vol_summary(novel_hash: str, vol_num: int) -> str:
        if storage is None:
            return _dump({"found": False, "reason": "no_storage_attached"})
        data = storage.get(f"{novel_hash}.summary.vol_{int(vol_num)}")
        if not data:
            return _dump({
                "found": False,
                "novel_hash": novel_hash,
                "vol_num": vol_num,
                "reason": "summary_not_generated",
            })
        return _dump({
            "found": True,
            "novel_hash": data.get("novel_hash", novel_hash),
            "vol_num": data.get("vol_num", vol_num),
            "summary": data.get("summary") or {},
        })

    tools = [world_tool, scene_tool, scene_content_tool]

    if storage is not None:
        vol_summary_tool = StructuredTool.from_function(
            func=_get_vol_summary,
            name="get_vol_summary",
            description=(
                "取某部小說某卷的 2-pass 卷摘要（主題 / 主角 / 主要角色 / 地點 / 大綱 / 主線 / 未解伏筆）。"
                "適合 meta 類問題：『這本書在講什麼』『主角是誰』『主要角色』『劇情大綱』等 —— "
                "比 search_world_knowledge 更快、更聚焦。找不到會回 found=false，此時改走 search_* 組合。"
                "回傳 YAML 格式。"
            ),
            args_schema=GetVolSummaryArgs,
        )
        tools.append(vol_summary_tool)

    return tools
