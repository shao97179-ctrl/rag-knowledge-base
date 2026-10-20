from pathlib import Path

from rag_learning.vector_store import MilvusStore

DIM = 8


def _vector(seed: float) -> list[float]:
    """生成一个 one-hot 向量，方便断言检索命中哪一行。"""
    vector = [0.0] * DIM
    vector[int(seed) % DIM] = 1.0
    return vector


def _rows() -> list[dict]:
    """4 行测试数据，覆盖两个部门、三个密级，用于验证 filter。"""
    base = {
        "chunk_index": 0,
        "content_hash": "hash",
        "doc_format": "markdown",
        "title": "标题",
        "heading": "章节",
    }
    return [
        {
            **base,
            "pk": "pk-ops",
            "doc_id": "d-ops",
            "parent_id": "p-ops",
            "child_text": "连接池耗尽",
            "parent_text": "连接池耗尽时先看 max_connections",
            "source_path": "ops/internal/pool.md",
            "department": "ops",
            "sensitivity": "internal",
            "dense": _vector(0),
            "sparse": {10: 1.0, 11: 0.5},
        },
        {
            **base,
            "pk": "pk-ops-2",
            "doc_id": "d-ops",
            "chunk_index": 1,
            "parent_id": "p-ops-2",
            "child_text": "慢查询排查",
            "parent_text": "按 total_exec_time 排序",
            "source_path": "ops/internal/pool.md",
            "department": "ops",
            "sensitivity": "internal",
            "dense": _vector(3),
            "sparse": {12: 1.0},
        },
        {
            **base,
            "pk": "pk-hr",
            "doc_id": "d-hr",
            "parent_id": "p-hr",
            "child_text": "薪酬带宽",
            "parent_text": "P7 月薪区间属于机密",
            "source_path": "hr/confidential/salary.md",
            "department": "hr",
            "sensitivity": "confidential",
            "dense": _vector(1),
            "sparse": {20: 1.0},
        },
        {
            **base,
            "pk": "pk-general",
            "doc_id": "d-general",
            "parent_id": "p-general",
            "child_text": "员工手册",
            "parent_text": "公司通用制度",
            "source_path": "general/public/handbook.md",
            "department": "general",
            "sensitivity": "public",
            "dense": _vector(2),
            "sparse": {30: 1.0},
        },
    ]


def test_store_upsert_search_filter_and_delete(tmp_path: Path) -> None:
    """存储层核心行为：写入计数、ACL 过滤、尾巴删除、按文档删除、重建。"""
    store = MilvusStore(uri=str(tmp_path / "milvus.db"), collection="test_chunks", dimension=DIM)
    try:
        store.ensure_collection()
        store.upsert(_rows())
        assert store.count() == 4

        # ops/internal 只能看到 ops 和 general/public，看不到 hr/confidential
        expression = 'department in ["ops", "general"] and sensitivity in ["public", "internal"]'
        dense_hits = store.search_dense(_vector(0), expression, limit=10)
        assert {hit["doc_id"] for hit in dense_hits} == {"d-ops", "d-general"}
        assert "d-hr" not in {hit["doc_id"] for hit in dense_hits}

        sparse_hits = store.search_sparse({10: 1.0}, expression, limit=10)
        assert sparse_hits[0]["doc_id"] == "d-ops"

        # 保留第 0 块，删掉 >= 1 的块
        store.delete_tail("d-ops", 1)
        assert store.count() == 3

        store.delete_doc("d-general")
        assert store.count() == 2

        store.recreate()
        assert store.count() == 0
        store.upsert(_rows())
        assert store.count() == 4
    finally:
        store.close()
