"""按扩展名分流的文档加载入口。"""

from __future__ import annotations

from pathlib import Path

from ..schema import ParsedDocument
from . import html_loader, office_loader, pdf_loader, tabular_loader, text_loader
from .base import (
    SUPPORTED_EXTENSIONS,
    AclOverride,
    ParseError,
    discover_documents,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "AclOverride",
    "ParseError",
    "discover_documents",
    "load_document",
]


def load_document(
    path: Path,
    docs_root: Path,
    override: AclOverride | None = None,
) -> ParsedDocument:
    """按扩展名选择解析器；解析异常统一包装成 ParseError。"""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ParseError(f"不支持的格式：{suffix}（{path}）")
    try:
        if suffix in {".md", ".markdown", ".txt"}:
            return text_loader.load(path, docs_root, override)
        if suffix in {".html", ".htm"}:
            return html_loader.load(path, docs_root, override)
        if suffix in {".csv", ".xlsx", ".xlsm"}:
            return tabular_loader.load(path, docs_root, override)
        if suffix in {".docx", ".pptx"}:
            return office_loader.load(path, docs_root, override)
        if suffix == ".pdf":
            return pdf_loader.load(path, docs_root, override)
    except ParseError:
        # 已经是我们自己的错误类型，原样抛出，避免重复包裹
        raise
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"解析失败：{path}：{exc}") from exc
    raise ParseError(f"没有对应的解析器：{suffix}")
