"""loader 公共工具：哈希、frontmatter、ACL 推断、表格转 Markdown。"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..cleaner import clean_text
from ..schema import DEPARTMENTS, SENSITIVITY_LEVELS, ParsedDocument

logger = logging.getLogger(__name__)

# 扩展名到内部格式名的映射；discover_documents 和 load_document 共用
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".docx": "docx",
    ".pptx": "pptx",
    ".pdf": "pdf",
}


class ParseError(Exception):
    """单文件解析失败；ingest 会记录 ERROR 并继续处理其他文件。"""


@dataclass
class AclOverride:
    """CLI 传入的权限覆盖，优先级最高。"""

    department: str | None = None
    sensitivity: str | None = None


def file_content_hash(path: Path) -> str:
    """文件 sha256，分块读取避免大文件占内存；用于增量判断。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def doc_id_for(rel_path: str) -> str:
    """由相对路径派生的稳定 doc_id；路径不变 id 就不变。"""
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]


def read_text_file(path: Path) -> str:
    """按 UTF-8-BOM -> GB18030 顺序尝试解码，最后用替换字符兜底。"""
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """解析 Markdown/TXT 文件头的 YAML；解析失败时退回原文。"""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    # frontmatter 以第二行 --- 或 ... 结束
    end = next((index for index in range(1, len(lines)) if lines[index].strip() in {"---", "..."}), None)
    if end is None:
        return {}, text
    try:
        data = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        logger.warning("frontmatter YAML 解析失败，忽略：%s", exc)
        return {}, text
    if not isinstance(data, dict):
        return {}, text
    return data, "\n".join(lines[end + 1:])


def _clean_meta(value: Any) -> str | None:
    """把 frontmatter 里的值规整成小写字符串，空值返回 None。"""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def infer_acl(
    rel_path: Path,
    frontmatter: dict[str, Any],
    override: AclOverride | None = None,
) -> tuple[str, str, str]:
    """权限优先级：CLI 覆盖 > frontmatter > 目录结构 > 默认值。"""
    source = "default"
    department: str | None = None
    sensitivity: str | None = None

    fm_department = _clean_meta(frontmatter.get("department"))
    fm_sensitivity = _clean_meta(frontmatter.get("sensitivity"))
    if fm_department or fm_sensitivity:
        source = "frontmatter"
        department = fm_department
        sensitivity = fm_sensitivity

    # 目录约定：docs/<部门>/<密级>/xxx，缺哪一项就由下一级来源补
    parts = [part.lower() for part in rel_path.parts]
    if department is None and len(parts) >= 2 and parts[0] in DEPARTMENTS:
        department = parts[0]
        source = source if source != "default" else "directory"
    if sensitivity is None and len(parts) >= 2 and parts[1] in SENSITIVITY_LEVELS:
        sensitivity = parts[1]
        source = source if source != "default" else "directory"

    if override and override.department:
        department = override.department
        source = "override"
    if override and override.sensitivity:
        sensitivity = override.sensitivity
        source = "override"

    # 兜底默认值：全公司可见的公开文档
    department = department or "general"
    sensitivity = sensitivity or "public"
    if department not in DEPARTMENTS:
        raise ParseError(f"未知部门 {department!r}：{rel_path}")
    if sensitivity not in SENSITIVITY_LEVELS:
        raise ParseError(f"未知密级 {sensitivity!r}：{rel_path}")
    return department, sensitivity, source


def first_heading(text: str) -> str | None:
    """取正文里第一个 Markdown 标题作为文档标题。"""
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return None


def build_document(
    path: Path,
    docs_root: Path,
    text: str,
    doc_format: str,
    frontmatter: dict[str, Any] | None = None,
    override: AclOverride | None = None,
) -> ParsedDocument:
    """各 loader 的公共收尾：算相对路径、清洗正文、推断权限、算 hash。"""
    frontmatter = frontmatter or {}
    try:
        rel_path = path.resolve().relative_to(docs_root.resolve())
    except ValueError:
        # 文件不在 docs 目录下（单文件入库）时只保留文件名
        rel_path = Path(path.name)
    body = clean_text(text)
    department, sensitivity, acl_source = infer_acl(rel_path, frontmatter, override)
    # 标题优先级：frontmatter title > 首个 Markdown 标题 > 文件名
    title = str(frontmatter.get("title") or first_heading(body) or path.stem).strip()
    return ParsedDocument(
        doc_id=doc_id_for(rel_path.as_posix()),
        source_path=path.resolve(),
        rel_path=rel_path.as_posix(),
        title=title[:200],
        doc_format=doc_format,
        text=body,
        department=department,
        sensitivity=sensitivity,
        content_hash=file_content_hash(path),
        metadata={"acl_source": acl_source, "frontmatter": frontmatter},
    )


def escape_cell(value: Any) -> str:
    """转义 Markdown 表格单元格：竖线要转义，换行改成 <br>。"""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>").strip()


def rows_to_markdown_table(rows: Iterable[Iterable[Any]], header: list[Any] | None = None) -> str:
    """把二维数据转成 Markdown 表格，自动补齐列宽和缺失单元格。"""
    normalized: list[list[str]] = []
    for row in rows:
        cells = [escape_cell(cell) for cell in row]
        # 整行为空的行直接丢掉，避免表格里出现大量空行
        if any(cells):
            normalized.append(cells)
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    if header is None:
        # 没显式给表头时，默认第一行是表头
        header_cells = normalized[0]
        body = normalized[1:]
    else:
        header_cells = [escape_cell(cell) for cell in header]
        body = normalized
    header_cells = header_cells + [""] * (width - len(header_cells))
    lines = ["| " + " | ".join(header_cells) + " |", "| " + " | ".join("---" for _ in range(width)) + " |"]
    for row in body:
        # 短行补空单元格，保证每行列数一致
        row = row + [""] * (width - len(row))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def dataframe_to_markdown(frame) -> str:
    """pandas DataFrame 转 Markdown 表格，NaN 统一成空字符串。"""
    header = [str(column) for column in frame.columns]
    body = frame.astype(object).where(frame.notna(), "").values.tolist()
    return rows_to_markdown_table(body, header=header)


def discover_documents(docs_root: Path) -> list[Path]:
    """递归扫描 docs 目录，返回所有受支持格式的文件（跳过隐藏目录）。"""
    if not docs_root.exists():
        return []
    files: list[Path] = []
    for path in sorted(docs_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # .git、.ipynb_checkpoints、__pycache__ 之类的目录一律跳过
        if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(docs_root).parts):
            continue
        files.append(path)
    return files
