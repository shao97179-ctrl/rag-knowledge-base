"""PDF 解析：只处理数字版 PDF；扫描件明确报错，不静默入库空文本。"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from pypdf import PdfReader

from ..schema import ParsedDocument
from .base import AclOverride, ParseError, build_document


def _pdf_title(path: Path) -> str | None:
    """从 PDF 元数据里读标题；读不到或解析失败就返回 None。"""
    try:
        metadata = PdfReader(str(path)).metadata
        title = (metadata.title or "").strip() if metadata else ""
        return title or None
    except Exception:  # noqa: BLE001
        return None



def load(path: Path, docs_root: Path, override: AclOverride | None = None) -> ParsedDocument:
    """逐页提取文本并加上页码标题，文本过少时判定为扫描件并报错。"""
    document = pdfium.PdfDocument(str(path))
    sections: list[str] = []
    total_chars = 0
    try:
        for page_number in range(len(document)):
            page = document[page_number]
            text_page = page.get_textpage()
            text = text_page.get_text_range()
            # pypdfium2 的对象需要手动释放，否则大 PDF 会累积内存
            text_page.close()
            page.close()
            total_chars += len(text.strip())
            # 页码转成二级标题，让 chunker 能按页维护 heading 路径
            sections.append(f"## 第 {page_number + 1} 页\n\n{text}")
    finally:
        document.close()

    # 数字版 PDF 正常都有几百字以上；过少基本是扫描件或空页
    if total_chars < 30:
        raise ParseError(f"PDF 可提取文本过少，可能是扫描件：{path.name}")
    title = _pdf_title(path)
    frontmatter = {"title": title} if title else {}
    return build_document(path, docs_root, "\n\n".join(sections), "pdf", frontmatter, override)
