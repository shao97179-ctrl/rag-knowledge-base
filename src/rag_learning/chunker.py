"""切分策略：先保护表格原子单元，再做父块 / 子块切分。

顺序很关键：如果先按长度切父块，超长表格会被硬切；所以这里先按行扫描，
把连续表格行连同最近的标题整体摘出来，表格不参与长度切分。
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schema import Chunk

logger = logging.getLogger(__name__)

# Markdown 标题行，例如 "## 数据库连接池"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# 表格数据行：整行以竖线包围
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
# 表格分隔行：| --- | :---: |
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

# 单个表格转成字符串后允许的最大字节数，超了就按行降级切分
MAX_TABLE_BYTES = 60_000
# 父块之间保留的字符重叠，避免句子被切在边界上
PARENT_OVERLAP = 100

# 优先按段落 / 换行 / 中文标点 / 空格切，最后才退化为按字符切
TEXT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


@dataclass
class Block:
    """正文中的一个逻辑块：普通文本或表格。"""

    kind: str  # text | table
    text: str
    heading: str


@dataclass
class ParentPiece:
    """父块中间表示，parent_index 决定 parent_id。"""

    kind: str
    text: str
    heading: str
    parent_index: int


def compute_pk(doc_id: str, chunk_index: int) -> str:
    """确定性主键：同一文档同一块永远不会变。"""
    return hashlib.sha1(f"{doc_id}#{chunk_index}".encode("utf-8")).hexdigest()


def compute_parent_id(doc_id: str, parent_index: int) -> str:
    """父块 ID：用 #p 前缀与子块主键区分，避免两套编号撞车。"""
    return hashlib.sha1(f"{doc_id}#p{parent_index}".encode("utf-8")).hexdigest()


def _heading_path(stack: list[tuple[int, str]]) -> str:
    """把标题栈拼成 "一级 > 二级 > 三级" 的路径字符串。"""
    return " > ".join(title for _, title in stack)


def _is_table_start(lines: list[str], index: int) -> bool:
    """判断当前行是否是一张表格的起点。

    优先看下一行是否是分隔行；没有分隔行的纯竖线块，要求至少连续两行，
    避免把正文里单独一行以竖线开头的内容误判成表格。
    """
    if not TABLE_LINE_RE.match(lines[index]):
        return False
    if index + 1 < len(lines) and TABLE_SEP_RE.match(lines[index + 1]):
        return True
    # 没有分隔行的纯竖线块，至少连续两行才当表格
    count = 0
    cursor = index
    while cursor < len(lines) and TABLE_LINE_RE.match(lines[cursor]):
        count += 1
        cursor += 1
    return count >= 2


def extract_blocks(text: str) -> list[Block]:
    """把正文拆成「文本块 / 表格块」，并维护标题路径。"""
    lines = text.split("\n")
    blocks: list[Block] = []
    buffer: list[str] = []
    # (标题级别, 标题文字) 栈，用于生成当前块所属的标题路径
    stack: list[tuple[int, str]] = []

    def flush() -> None:
        """把缓冲区里的文本行落成一个文本块，并清空缓冲区。"""
        if buffer:
            content = "\n".join(buffer).strip()
            if content:
                blocks.append(Block("text", content, _heading_path(stack)))
            buffer.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        heading = HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            # 同级或更高级标题出现时弹出栈，保证栈里始终是当前路径
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buffer.append(line)
            index += 1
            continue
        if _is_table_start(lines, index):
            flush()
            rows: list[str] = []
            # 连续表格行整体摘出，作为原子块，不参与后续长度切分
            while index < len(lines) and TABLE_LINE_RE.match(lines[index]):
                rows.append(lines[index].strip())
                index += 1
            blocks.append(Block("table", "\n".join(rows), _heading_path(stack)))
            continue
        buffer.append(line)
        index += 1
    flush()

    # 相邻文本块合并，减少碎片；表格块不参与合并
    merged: list[Block] = []
    for block in blocks:
        if merged and block.kind == "text" and merged[-1].kind == "text":
            merged[-1].text = f"{merged[-1].text}\n{block.text}"
        else:
            merged.append(block)

    result: list[Block] = []
    for block in merged:
        if block.kind == "text":
            # 去掉纯标题行后如果正文不足 5 个字符，视为空块直接丢弃
            body = "\n".join(
                line for line in block.text.split("\n") if not HEADING_RE.match(line)
            ).strip()
            if len(body) < 5:
                continue
        result.append(block)
    return result


def _split_large_table(text: str) -> list[str]:
    """超过 Milvus VARCHAR 上限的表格按行降级切分，并打 WARNING。"""
    if len(text.encode("utf-8")) <= MAX_TABLE_BYTES:
        return [text]

    logger.warning("表格超过 %s 字节，按行降级切分", MAX_TABLE_BYTES)
    lines = text.split("\n")
    # 表头（含分隔行）会在每个分组里重复，保证每块都是合法 Markdown 表格
    header = lines[:2] if len(lines) >= 2 and TABLE_SEP_RE.match(lines[1]) else []
    body = lines[len(header):]
    header_size = len("\n".join(header).encode("utf-8"))
    groups: list[str] = []
    current: list[str] = []
    current_size = 0

    for row in body:
        row_size = len(row.encode("utf-8")) + 1
        if current and header_size + current_size + row_size > MAX_TABLE_BYTES:
            groups.append("\n".join(header + current))
            current = []
            current_size = 0
        # 单行本身就超上限时无法再按行切，只能硬切成 1 万字符的片段
        if row_size > MAX_TABLE_BYTES - header_size:
            logger.warning("单行表格超过上限，执行硬切")
            for start in range(0, len(row), 10_000):
                groups.append("\n".join(header + [row[start:start + 10_000]]))
            continue
        current.append(row)
        current_size += row_size
    if current:
        groups.append("\n".join(header + current))
    return groups or [text]


def _parent_pieces(blocks: list[Block], parent_size: int) -> list[ParentPiece]:
    """把文本块按 parent_size 切成父块；表格块原样成为独立父块。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=PARENT_OVERLAP,
        separators=TEXT_SEPARATORS,
        keep_separator=True,
    )
    pieces: list[ParentPiece] = []
    for block in blocks:
        if block.kind == "table":
            # 表格不参与长度切分，只有超过字节上限时才降级拆行
            for table in _split_large_table(block.text):
                pieces.append(ParentPiece("table", table, block.heading, len(pieces)))
            continue
        for part in splitter.split_text(block.text):
            part = part.strip()
            if part:
                pieces.append(ParentPiece("text", part, block.heading, len(pieces)))
    return pieces


def chunk_document(
    text: str,
    doc_id: str,
    parent_size: int = 1200,
    child_size: int = 350,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """正文 -> 父块 -> 子块。表格父块和子块都保持整体。"""
    blocks = extract_blocks(text)
    pieces = _parent_pieces(blocks, parent_size)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=chunk_overlap,
        separators=TEXT_SEPARATORS,
        keep_separator=True,
    )

    chunks: list[Chunk] = []
    for piece in pieces:
        parent_id = compute_parent_id(doc_id, piece.parent_index)
        if piece.kind == "table":
            # 表格整体作为唯一子块，保证进索引的文本结构完整
            child_texts = [piece.text]
        else:
            # 极端情况下 splitter 返回空列表，回退成整个父块
            child_texts = child_splitter.split_text(piece.text) or [piece.text]
        for child in child_texts:
            child = child.strip()
            if not child:
                continue
            # 标题前缀让子块脱离父块后仍能保留一点上下文
            prefix = f"【{piece.heading}】\n" if piece.heading else ""
            # chunk_index 全局连续，配合确定性主键保证 upsert 幂等
            chunk_index = len(chunks)
            chunks.append(
                Chunk(
                    pk=compute_pk(doc_id, chunk_index),
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    parent_id=parent_id,
                    parent_index=piece.parent_index,
                    child_text=prefix + child,
                    parent_text=piece.text,
                    heading=piece.heading,
                )
            )
    return chunks
