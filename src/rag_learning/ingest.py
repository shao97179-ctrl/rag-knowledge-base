"""入库管道：解析 -> 清洗 -> 切分 -> 编码 -> upsert，支持增量与 force 重建。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunker import chunk_document
from .config import PROJECT_ROOT
from .loaders import AclOverride, ParseError, discover_documents, load_document
from .registry import DocumentRecord, Registry
from .schema import ParsedDocument
from .vector_store import MilvusStore

logger = logging.getLogger(__name__)

# 落库前的字节上限，比 Milvus schema 的 max_length 略小，留出安全余量
CHILD_MAX_BYTES = 8000
PARENT_MAX_BYTES = 65_000
HEADING_MAX_CHARS = 500
TITLE_MAX_BYTES = 500


def truncate_utf8(text: str, max_bytes: int) -> str:
    """按 UTF-8 字节数截断，不会切出半个汉字（errors="ignore"）。"""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


@dataclass
class IngestResult:
    """单个文件的入库结果。"""

    rel_path: str
    status: str  # indexed | unchanged | deleted | empty | failed
    chunks: int = 0
    doc_format: str = ""
    department: str = ""
    sensitivity: str = ""
    error: str | None = None


@dataclass
class IngestSummary:
    """一次入库的汇总，按状态计数。"""

    results: list[IngestResult] = field(default_factory=list)

    @property
    def indexed(self) -> int:
        """本次真正写入索引的文件数。"""
        return sum(1 for result in self.results if result.status == "indexed")

    @property
    def unchanged(self) -> int:
        """内容未变化、被跳过的文件数。"""
        return sum(1 for result in self.results if result.status == "unchanged")

    @property
    def deleted(self) -> int:
        """已从磁盘删除、本次清理索引的文件数。"""
        return sum(1 for result in self.results if result.status == "deleted")

    @property
    def failed(self) -> int:
        """解析或入库失败的文件数。"""
        return sum(1 for result in self.results if result.status == "failed")

    @property
    def empty(self) -> int:
        """没有可用文本的文件数。"""
        return sum(1 for result in self.results if result.status == "empty")

    @property
    def total_chunks(self) -> int:
        """当前索引中属于本次处理文档的 chunk 总数。"""
        return sum(result.chunks for result in self.results if result.status in {"indexed", "unchanged"})


class Ingestor:
    """编排单个或全量文档的入库流程。"""

    def __init__(
        self,
        settings: Any,
        embedder: Any,
        store: MilvusStore,
        registry: Registry,
    ) -> None:
        """依赖注入，测试里可以传入 FakeEmbedder 和临时目录。"""
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.registry = registry

    def ingest(
        self,
        paths: list[Path] | None = None,
        force: bool = False,
        override: AclOverride | None = None,
    ) -> IngestSummary:
        """入库主流程：可选全量扫描、force 重建、单文件处理以及删除清理。"""
        if force and paths:
            # force 会先清空集合和注册表，若此时只处理指定文件，
            # 整个索引会被替换成这一个文件，属于必须拦下的误操作
            raise ValueError(
                "--force 会清空整个索引后重建，不能与指定文件同时使用；"
                "需要重建单篇文档请直接全量 ingest --force，或去掉 --force 单独入库该文件"
            )
        docs_root: Path = self.settings.docs_dir
        summary = IngestSummary()

        if force:
            # 重建前先清空向量库和注册表，保证不残留旧块
            logger.info("--force：重建集合和注册表")
            self.store.recreate()
            self.registry.clear()

        if paths:
            targets = [self._resolve(path) for path in paths]
            # 指定单文件时不扫描全库，也就不做删除清理
            full_scan = False
        else:
            targets = discover_documents(docs_root)
            full_scan = True

        logger.info("待处理文档：%s 个", len(targets))
        for target in targets:
            summary.results.append(self._process(target, docs_root, force, override))

        if full_scan:
            # 全量扫描时，注册表里有但磁盘上已经没有的文档要清掉索引
            discovered = {target.resolve() for target in targets}
            for record in self.registry.list_all():
                record_path = (docs_root / record.rel_path).resolve()
                if record_path in discovered:
                    continue
                self.store.delete_doc(record.doc_id)
                self.registry.delete(record.doc_id)
                logger.info("文档已删除，清理索引：%s（%s 块）", record.rel_path, record.chunk_count)
                summary.results.append(
                    IngestResult(rel_path=record.rel_path, status="deleted", chunks=record.chunk_count)
                )

        logger.info(
            "入库汇总：indexed=%s unchanged=%s deleted=%s empty=%s failed=%s chunks=%s",
            summary.indexed,
            summary.unchanged,
            summary.deleted,
            summary.empty,
            summary.failed,
            summary.total_chunks,
        )
        return summary

    def _resolve(self, path: Path) -> Path:
        """把用户传入的路径解析成绝对路径：先试项目根，再试当前工作目录。"""
        path = path.expanduser()
        if not path.is_absolute():
            candidate = (PROJECT_ROOT / path).resolve()
            path = candidate if candidate.exists() else (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        if not path.is_file():
            raise ValueError(f"不是文件：{path}")
        return path

    def _process(
        self,
        path: Path,
        docs_root: Path,
        force: bool,
        override: AclOverride | None,
    ) -> IngestResult:
        """解析单个文件并入库；任何异常都收敛成 failed 结果，不中断整批。"""
        try:
            document = load_document(path, docs_root, override)
        except ParseError as exc:
            logger.error("解析失败：%s：%s", path, exc)
            return IngestResult(rel_path=self._rel(path, docs_root), status="failed", error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("加载异常：%s", path)
            return IngestResult(rel_path=self._rel(path, docs_root), status="failed", error=str(exc))

        try:
            return self._index_document(document, force)
        except Exception as exc:  # noqa: BLE001
            logger.exception("入库失败：%s", document.rel_path)
            return IngestResult(
                rel_path=document.rel_path,
                status="failed",
                doc_format=document.doc_format,
                department=document.department,
                sensitivity=document.sensitivity,
                error=str(exc),
            )

    def _index_document(self, document: ParsedDocument, force: bool) -> IngestResult:
        """对已解析的文档做增量判断、切分、编码和写入。"""
        existing = self.registry.get(document.doc_id)
        # 内容 hash 和权限都没变才跳过；权限变了要重新写，让 filter 立即生效
        if (
            not force
            and existing is not None
            and existing.status == "indexed"
            and existing.content_hash == document.content_hash
            and existing.department == document.department
            and existing.sensitivity == document.sensitivity
        ):
            logger.info("内容未变化，跳过：%s", document.rel_path)
            return IngestResult(
                rel_path=document.rel_path,
                status="unchanged",
                chunks=existing.chunk_count,
                doc_format=document.doc_format,
                department=document.department,
                sensitivity=document.sensitivity,
            )

        chunks = chunk_document(
            document.text,
            document.doc_id,
            parent_size=self.settings.parent_size,
            child_size=self.settings.child_size,
            chunk_overlap=self.settings.chunk_overlap,
            parent_overlap=self.settings.parent_overlap,
        )
        if not chunks:
            # 解析出空文本：清掉旧索引，但保留注册表记录（status=empty）
            self.store.delete_doc(document.doc_id)
            self.registry.upsert(
                DocumentRecord(
                    doc_id=document.doc_id,
                    rel_path=document.rel_path,
                    content_hash=document.content_hash,
                    department=document.department,
                    sensitivity=document.sensitivity,
                    doc_format=document.doc_format,
                    chunk_count=0,
                    indexed_at=datetime.now(timezone.utc).isoformat(),
                    status="empty",
                )
            )
            logger.warning("文档没有可用文本：%s", document.rel_path)
            return IngestResult(
                rel_path=document.rel_path,
                status="empty",
                doc_format=document.doc_format,
                department=document.department,
                sensitivity=document.sensitivity,
            )

        dense_vectors, sparse_vectors = self.embedder.encode([chunk.child_text for chunk in chunks])
        rows = [
            self._build_row(document, chunk, dense, sparse)
            for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
        ]

        if existing is not None and existing.chunk_count > len(chunks):
            # 文档变短：先删掉多余的尾部块，再 upsert，顺序不能反
            logger.info(
                "文档变短，先删除尾巴：%s（%s -> %s）",
                document.rel_path,
                existing.chunk_count,
                len(chunks),
            )
            self.store.delete_tail(document.doc_id, len(chunks))

        self.store.upsert(rows)
        self.registry.upsert(
            DocumentRecord(
                doc_id=document.doc_id,
                rel_path=document.rel_path,
                content_hash=document.content_hash,
                department=document.department,
                sensitivity=document.sensitivity,
                doc_format=document.doc_format,
                chunk_count=len(chunks),
                indexed_at=datetime.now(timezone.utc).isoformat(),
                status="indexed",
            )
        )
        logger.info(
            "入库完成：%s（%s，%s/%s，%s 块）",
            document.rel_path,
            document.doc_format,
            document.department,
            document.sensitivity,
            len(chunks),
        )
        return IngestResult(
            rel_path=document.rel_path,
            status="indexed",
            chunks=len(chunks),
            doc_format=document.doc_format,
            department=document.department,
            sensitivity=document.sensitivity,
        )

    def _build_row(self, document: ParsedDocument, chunk, dense: list[float], sparse: dict[int, float]) -> dict:
        """把一个 Chunk 组装成 Milvus 行，并做字节级截断保护。"""
        child_text = truncate_utf8(chunk.child_text, CHILD_MAX_BYTES)
        parent_text = truncate_utf8(chunk.parent_text, PARENT_MAX_BYTES)
        if len(child_text) != len(chunk.child_text) or len(parent_text) != len(chunk.parent_text):
            logger.warning("块超过 Milvus VARCHAR 上限，已截断：%s#%s", document.rel_path, chunk.chunk_index)
        return {
            "pk": chunk.pk,
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "parent_id": chunk.parent_id,
            "child_text": child_text,
            "parent_text": parent_text,
            "heading": chunk.heading[:HEADING_MAX_CHARS],
            "title": truncate_utf8(document.title, TITLE_MAX_BYTES),
            "source_path": document.rel_path,
            "department": document.department,
            "sensitivity": document.sensitivity,
            "doc_format": document.doc_format,
            "content_hash": document.content_hash,
            "dense": dense,
            "sparse": sparse,
        }

    @staticmethod
    def _rel(path: Path, docs_root: Path) -> str:
        """尽量返回相对 docs 根目录的路径；文件在 docs 外时退回文件名。"""
        try:
            return path.resolve().relative_to(docs_root.resolve()).as_posix()
        except ValueError:
            return path.name
