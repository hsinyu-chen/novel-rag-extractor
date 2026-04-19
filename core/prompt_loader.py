import re
from functools import lru_cache
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


@lru_cache(maxsize=None)
def _load_raw(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip("\n")


def load_prompt(name: str) -> str:
    """載入 prompt 檔案內容（移除 front-matter）。"""
    return _load_raw(name)


def render_prompt(name: str, **vars) -> str:
    """載入 prompt 並以 string.Template 的 $var 語法做變數插值。

    使用 $var 而非 {var} 以避免與 JSON 範例、Markdown 排版衝突。
    """
    text = _load_raw(name)
    if not vars:
        return text
    return Template(text).substitute(**vars)
