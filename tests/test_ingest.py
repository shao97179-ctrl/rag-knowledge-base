from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_learning.ingest import Ingestor
from rag_learning.registry import Registry
from rag_learning.vector_store import MilvusStore

DIM = 8


class FakeEmbedder:
    """测试用假编码器，不加载 BGE-M3。"""

    DIMENSION = DIM

    def encode(self, texts: list[str], is_query: bool = False):
        """按字符哈希生成确定性的 8 维 dense 和稀疏词权重，只为跑通链路。"""
        dense = []
        sparse = []
        for text in texts:
            vector = [0.0] * DIM
            for index, char in enumerate(text[:20]):
                vector[(ord(char) + index) % DIM] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            dense.append([value / norm for value in vector])
            weights: dict[int, float] = {}
            for char in text[:10]:
                token = ord(char) % 500 + 1
                weights[token] = weights.get(token, 0.0) + 1.0
            # 空稀疏向量 Milvus 不接受，用占位值
            sparse.append(weights or {0: 1e-7})
        return dense, sparse


def _long_text(paragraphs: int) -> str:
    """生成指定段落数的连接池文档，用来模拟文档变长 / 变短。"""
    body = "\n\n".join(
        f"第 {index} 段：连接池耗尽时先检查 max_connections、慢查询和连接泄漏。" for index in range(paragraphs)
    )
    return f"# 连接池排查\n\n{body}"


def _setup(tmp_path: Path):
    """建一个最小 docs 目录（ops/internal + hr/confidential）和配套组件。"""
    docs = tmp_path / "docs"
    (docs / "ops/internal").mkdir(parents=True)
    (docs / "hr/confidential").mkdir(parents=True)
    pool = docs / "ops/internal/pool.md"
    salary = docs / "hr/confidential/salary.md"
    pool.write_text(_long_text(30), encoding="utf-8")
    salary.write_text("# 薪酬带宽\n\nP7 月薪区间为 45000 到 62000 元。", encoding="utf-8")

    settings = SimpleNamespace(
        docs_dir=docs,
        parent_size=300,
        child_size=80,
        chunk_overlap=20,
    )
    store = MilvusStore(uri=str(tmp_path / "milvus.db"), collection="test_chunks", dimension=DIM)
    registry = Registry(tmp_path / "registry.sqlite")
    return pool, salary, Ingestor(settings, FakeEmbedder(), store, registry), store, registry


def test_force_with_explicit_paths_is_rejected(tmp_path: Path) -> None:
    """force 与指定文件互斥：误操作必须在清库之前被拦下，索引原封不动。"""
    pool, _, ingestor, store, registry = _setup(tmp_path)
    try:
        first = ingestor.ingest()
        assert first.indexed == 2

        with pytest.raises(ValueError, match="--force"):
            ingestor.ingest(paths=[pool], force=True)

        # 守卫在 recreate 之前生效：集合和注册表都不能被动过
        assert store.count() == first.total_chunks
        assert len(registry.list_all()) == 2
    finally:
        store.close()


def test_incremental_ingest_and_acl(tmp_path: Path) -> None:
    """增量入库全流程：首次入库、重复跳过、变短删尾、force 重建、删除清理。"""
    pool, salary, ingestor, store, registry = _setup(tmp_path)
    try:
        first = ingestor.ingest()
        assert first.indexed == 2
        assert first.failed == 0
        assert first.total_chunks > 2

        pool_record = registry.get_by_path("ops/internal/pool.md")
        salary_record = registry.get_by_path("hr/confidential/salary.md")
        assert pool_record is not None and salary_record is not None
        assert pool_record.chunk_count > 1

        # 内容没变，第二次应该全部跳过
        second = ingestor.ingest()
        assert second.unchanged == 2
        assert second.indexed == 0

        # ops 用户检索不到 hr/confidential 的内容
        expression = 'department in ["ops", "general"] and sensitivity in ["public", "internal"]'
        hits = store.search_dense([1.0] + [0.0] * (DIM - 1), expression, limit=20)
        assert hits
        assert {hit["doc_id"] for hit in hits} == {pool_record.doc_id}

        # 文档变短：先删尾巴，再 upsert
        pool.write_text(_long_text(8), encoding="utf-8")
        third = ingestor.ingest()
        assert third.indexed == 1
        assert third.unchanged == 1
        new_count = registry.get_by_path("ops/internal/pool.md").chunk_count
        assert new_count < pool_record.chunk_count
        rows = store.client.query(
            collection_name="test_chunks",
            filter=f'doc_id == "{pool_record.doc_id}"',
            output_fields=["chunk_index"],
        )
        # 最大 chunk_index 必须等于新块数 - 1，说明旧尾巴确实被删掉了
        assert max(row["chunk_index"] for row in rows) == new_count - 1

        # force 重建
        forced = ingestor.ingest(force=True)
        assert forced.indexed == 2
        assert store.count() == sum(record.chunk_count for record in registry.list_all())

        # 删除文档：全量扫描时清理索引和注册表
        pool.unlink()
        deleted = ingestor.ingest()
        assert deleted.deleted == 1
        assert registry.get_by_path("ops/internal/pool.md") is None
        assert store.count() == salary_record.chunk_count
    finally:
        store.close()
