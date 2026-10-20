"""Markdown / 纯文本解析。"""

from __future__ import annotations

from pathlib import Path

from ..schema import ParsedDocument
from .base import AclOverride, build_document, read_text_file, split_frontmatter


def load(path: Path, docs_root: Path, override: AclOverride | None = None) -> ParsedDocument:
    """读文本、剥离 YAML frontmatter，然后走公共的清洗与权限推断。"""
    raw = read_text_file(path)
    frontmatter, body = split_frontmatter(raw)
    doc_format = "markdown" if path.suffix.lower() in {".md", ".markdown"} else "text"
    return build_document(path, docs_root, body, doc_format, frontmatter, override)
