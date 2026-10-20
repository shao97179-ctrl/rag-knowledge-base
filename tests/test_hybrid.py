from pathlib import Path
from types import SimpleNamespace

from rag_learning.retrieval.hybrid import HybridRetriever
from rag_learning.retrieval.query_rewrite import QueryRewriter
from rag_learning.schema import UserContext
from rag_learning.vector_store import MilvusStore

DIM = 8


class FakeQueryEmbedder:
    """查询固定返回同一个向量，保证两路都能命中测试数据。"""

    def encode_query(self, query: str):
        """返回固定 dense 向量和固定稀疏词，不依赖真实模型。"""
        return [1.0] + [0.0] * (DIM - 1), {1: 1.0}


def _row(pk: str, doc_id: str, parent_id: str, department: str, sensitivity: str, text: str, sparse: dict) -> dict:
    """构造一行向量库测试数据。"""
    return {
        "pk": pk,
        "doc_id": doc_id,
        "chunk_index": 0,
        "parent_id": parent_id,
        "child_text": text,
        "parent_text": text,
        "heading": "章节",
        "title": "标题",
        "source_path": f"{department}/{sensitivity}/{doc_id}.md",
        "department": department,
        "sensitivity": sensitivity,
        "doc_format": "markdown",
        "content_hash": "hash",
        "dense": [1.0] + [0.0] * (DIM - 1),
        "sparse": sparse,
    }


def test_hybrid_search_dedupes_parents_and_applies_acl(tmp_path: Path) -> None:
    """同一父块只保留一次；无权的 hr/confidential 不能出现在结果里。"""
    store = MilvusStore(uri=str(tmp_path / "milvus.db"), collection="hybrid", dimension=DIM)
    settings = SimpleNamespace(top_k_dense=10, top_k_sparse=10, top_k_final=5, rrf_k=60)
    try:
        store.upsert(
            [
                _row("pk1", "pool", "p1", "ops", "internal", "连接池耗尽应急处理", {1: 1.0}),
                _row("pk2", "pool", "p1", "ops", "internal", "连接池慢查询排查", {1: 0.5}),
                _row("pk3", "salary", "p3", "hr", "confidential", "薪酬带宽", {1: 0.9}),
                _row("pk4", "handbook", "p4", "general", "public", "员工手册", {2: 1.0}),
            ]
        )
        retriever = HybridRetriever(settings, FakeQueryEmbedder(), store)
        result = retriever.search("连接池耗尽", UserContext("ops", "internal"))

        # pk1 和 pk2 属于同一个父块 p1，去重后只剩一条
        assert len(result.contexts) == 2
        assert {context.parent_id for context in result.contexts} == {"p1", "p4"}
        assert all(context.department in {"ops", "general"} for context in result.contexts)
        assert result.debug["dense_count"] == 3
        assert result.debug["sparse_count"] == 2
        assert "department in" in result.debug["filter_expr"]
    finally:
        store.close()


def test_query_rewrite_only_with_history() -> None:
    """只有多轮且 LLM 可用时才改写；单轮和离线模式都原样返回。"""
    class FakeLLM:
        available = True

        def chat(self, system, user, model=None, temperature=None):
            """模拟模型只输出一行改写结果。"""
            return "连接池耗尽了应该怎么处理？"

    rewriter = QueryRewriter(FakeLLM(), "light")
    assert rewriter.rewrite("它怎么处理", [{"role": "user", "content": "连接池耗尽"}]) == (
        "连接池耗尽了应该怎么处理？"
    )
    # 单轮不改写
    assert rewriter.rewrite("单轮问题", None) == "单轮问题"

    class OfflineLLM:
        available = False

        def chat(self, *args, **kwargs):
            """离线模式不该被调用，调到就说明逻辑有 bug。"""
            raise AssertionError("离线模式不应调用 LLM")

    assert QueryRewriter(OfflineLLM(), "light").rewrite(
        "它呢", [{"role": "user", "content": "连接池"}]
    ) == "它呢"
