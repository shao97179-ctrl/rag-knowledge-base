from pathlib import Path

from rag_learning.registry import DocumentRecord, Registry


def _record(chunk_count: int = 3) -> DocumentRecord:
    """构造一条测试用文档记录。"""
    return DocumentRecord(
        doc_id="doc1",
        rel_path="ops/internal/pool.md",
        content_hash="hash1",
        department="ops",
        sensitivity="internal",
        doc_format="markdown",
        chunk_count=chunk_count,
        indexed_at="2026-10-20T00:00:00+00:00",
    )


def test_registry_crud(tmp_path: Path) -> None:
    """注册表增删改查：upsert 覆盖、按 id/路径查询、delete 和 clear。"""
    registry = Registry(tmp_path / "registry.sqlite")
    record = _record()
    registry.upsert(record)

    loaded = registry.get("doc1")
    assert loaded is not None
    assert loaded.rel_path == record.rel_path
    assert loaded.chunk_count == 3
    assert registry.get_by_path(record.rel_path).doc_id == "doc1"

    # 同一 doc_id 再次 upsert 是更新而不是插入
    registry.upsert(_record(chunk_count=5))
    assert registry.get("doc1").chunk_count == 5

    assert len(registry.list_all()) == 1
    registry.delete("doc1")
    assert registry.get("doc1") is None

    registry.upsert(_record())
    registry.clear()
    assert registry.list_all() == []
