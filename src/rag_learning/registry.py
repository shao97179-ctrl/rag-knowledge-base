"""SQLite 文档注册表：记录 content_hash、chunk 数、权限，用于增量入库。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class DocumentRecord:
    """documents 表的一行：文档级元数据，不存正文。"""

    doc_id: str
    rel_path: str
    content_hash: str
    department: str
    sensitivity: str
    doc_format: str
    chunk_count: int
    indexed_at: str
    status: str = "indexed"


class Registry:
    """文档注册表，所有 SQL 都在这个类里。"""

    def __init__(self, path: Path) -> None:
        """建好父目录和表结构，重复初始化是安全的。"""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def _connect(self) -> sqlite3.Connection:
        """每次操作新开连接（sqlite 连接不能跨线程共享），行按列名访问。"""
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def init(self) -> None:
        """建表；rel_path 唯一，避免同一文件被登记成两条记录。"""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    rel_path TEXT UNIQUE NOT NULL,
                    content_hash TEXT NOT NULL,
                    department TEXT NOT NULL,
                    sensitivity TEXT NOT NULL,
                    doc_format TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'indexed'
                )
                """
            )

    def get(self, doc_id: str) -> DocumentRecord | None:
        """按 doc_id 查一条记录，不存在返回 None。"""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def get_by_path(self, rel_path: str) -> DocumentRecord | None:
        """按相对路径查一条记录，主要用于测试和排查。"""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE rel_path = ?", (rel_path,)).fetchone()
        return self._row_to_record(row) if row else None

    def upsert(self, record: DocumentRecord) -> None:
        """按 doc_id 插入或更新；indexed_at 为空时补当前 UTC 时间。"""
        indexed_at = record.indexed_at or datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    doc_id, rel_path, content_hash, department, sensitivity,
                    doc_format, chunk_count, indexed_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    rel_path = excluded.rel_path,
                    content_hash = excluded.content_hash,
                    department = excluded.department,
                    sensitivity = excluded.sensitivity,
                    doc_format = excluded.doc_format,
                    chunk_count = excluded.chunk_count,
                    indexed_at = excluded.indexed_at,
                    status = excluded.status
                """,
                (
                    record.doc_id,
                    record.rel_path,
                    record.content_hash,
                    record.department,
                    record.sensitivity,
                    record.doc_format,
                    record.chunk_count,
                    indexed_at,
                    record.status,
                ),
            )

    def delete(self, doc_id: str) -> None:
        """删除单条记录，与向量库的 delete_doc 配套使用。"""
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def list_all(self) -> list[DocumentRecord]:
        """按相对路径排序列出全部文档，供 API / CLI 展示。"""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM documents ORDER BY rel_path").fetchall()
        return [self._row_to_record(row) for row in rows]

    def clear(self) -> None:
        """清空注册表，配合 --force 重建使用。"""
        with self._connect() as connection:
            connection.execute("DELETE FROM documents")

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        """sqlite3.Row 转 dataclass，集中处理列名到字段的映射。"""
        return DocumentRecord(
            doc_id=row["doc_id"],
            rel_path=row["rel_path"],
            content_hash=row["content_hash"],
            department=row["department"],
            sensitivity=row["sensitivity"],
            doc_format=row["doc_format"],
            chunk_count=row["chunk_count"],
            indexed_at=row["indexed_at"],
            status=row["status"],
        )
