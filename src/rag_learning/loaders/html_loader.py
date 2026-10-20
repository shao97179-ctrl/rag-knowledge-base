"""HTML 解析：去脚本样式、标题转 Markdown、表格保留结构。"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup

from ..schema import ParsedDocument
from .base import AclOverride, build_document, read_text_file, rows_to_markdown_table


def _table_to_markdown(table) -> str:
    """把 <table> 里的 th/td 抽成二维表，再转 Markdown 表格。"""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    return rows_to_markdown_table(rows)


def load(path: Path, docs_root: Path, override: AclOverride | None = None) -> ParsedDocument:
    """把 HTML 转成接近 Markdown 的纯文本，尽量保留标题层级和列表。"""
    soup = BeautifulSoup(read_text_file(path), "lxml")
    # 脚本、样式、SVG 等不承载正文，先整体删除
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    # 表格必须在 get_text 之前替换，否则单元格会被拼成一行
    for table in soup.find_all("table"):
        markdown = _table_to_markdown(table)
        if markdown:
            table.replace_with(soup.new_string(f"\n\n{markdown}\n\n"))
        else:
            # 空表格没有信息量，直接丢掉
            table.decompose()

    # h1-h6 转成 Markdown 等号标题，chunker 才能识别标题层级
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        level = int(heading.name[1])
        title = heading.get_text(" ", strip=True)
        heading.replace_with(soup.new_string(f"\n{'#' * level} {title}\n"))

    # 列表项前面补 "- "，保留列表语义
    for item in soup.find_all("li"):
        item.insert(0, "- ")

    # 块级元素后面补换行，避免所有段落粘成一段
    for block in soup.find_all(["p", "div", "section", "article", "br", "tr"]):
        block.append("\n")

    text = soup.get_text("\n")
    return build_document(path, docs_root, text, "html", {}, override)
