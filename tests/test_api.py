from types import SimpleNamespace

from fastapi.testclient import TestClient

from rag_learning.api import app, get_pipeline_dep
from rag_learning.registry import DocumentRecord
from rag_learning.retrieval.hybrid import RetrievalResult
from rag_learning.schema import AnswerResult, Citation, RetrievedChunk


def _chunk() -> RetrievedChunk:
    """构造一条固定的检索结果，供 search / chat 桩复用。"""
    return RetrievedChunk(
        pk="pk1",
        doc_id="doc1",
        chunk_index=0,
        parent_id="parent1",
        child_text="UPI 的单笔交易限额为 10 万卢比。",
        parent_text="UPI 的单笔交易限额为 10 万卢比，每日限额也是 10 万卢比。",
        heading="UPI",
        title="统一支付接口（UPI）指南",
        source_path="ops/internal/upi-zx-bank-asia.pdf",
        department="ops",
        sensitivity="internal",
        doc_format="text",
        score=0.0325,
        dense_rank=1,
        sparse_rank=1,
        dense_score=0.71,
        sparse_score=0.44,
    )


class FakeLLM:
    """离线 LLM 桩，让 health 返回 available=False。"""

    available = False


class FakeRegistry:
    """注册表桩：返回两条权限不同的记录，供 /documents 验证过滤。"""

    def list_all(self):
        """一条 ops/internal，一条 rd/confidential。"""
        return [
            DocumentRecord(
                doc_id="doc1",
                rel_path="ops/internal/upi-zx-bank-asia.pdf",
                content_hash="hash",
                department="ops",
                sensitivity="internal",
                doc_format="text",
                chunk_count=1,
                indexed_at="2026-10-20T00:00:00+00:00",
            ),
            DocumentRecord(
                doc_id="doc2",
                rel_path="rd/confidential/fraud-transaction.md",
                content_hash="hash",
                department="rd",
                sensitivity="confidential",
                doc_format="markdown",
                chunk_count=1,
                indexed_at="2026-10-20T00:00:00+00:00",
            ),
        ]


class FakeStore:
    """向量库桩：只提供 stats()。"""

    def stats(self):
        """返回假的集合状态。"""
        return {"rows": 1, "collection": "test", "exists": True}


class FakePipeline:
    """整条 pipeline 的桩，绕开 BGE-M3 和 Milvus。"""

    settings = SimpleNamespace(llm_provider="offline", llm_model="offline")
    llm = FakeLLM()
    registry = FakeRegistry()
    store = FakeStore()

    def health(self):
        """模拟 /health 的返回结构。"""
        return {"llm": {"available": False}, "documents": 1}

    def search(self, query, user, top_k=None):
        """模拟检索：固定返回一条上下文。"""
        return RetrievalResult(
            contexts=[_chunk()],
            debug={"filter_expr": "department in [\"ops\"]", "dense_count": 1},
        )

    def ask(self, query, user, history=None, top_k=None):
        """模拟问答：返回带引用的固定答案。"""
        return AnswerResult(
            answer="UPI 单笔限额为 10 万卢比 [1]。",
            citations=[
                Citation(
                    index=1,
                    title="统一支付接口（UPI）指南",
                    source_path="ops/internal/upi-zx-bank-asia.pdf",
                    heading="UPI",
                    department="ops",
                    sensitivity="internal",
                    snippet="UPI 的单笔交易限额为 10 万卢比。",
                    score=0.0325,
                )
            ],
            contexts=[_chunk()],
            rewritten_query=query,
            refused=False,
            offline=True,
            debug={"retrieval": {"dense_count": 1, "sparse_count": 1}},
        )


def _client() -> TestClient:
    """用 dependency_overrides 把真实 pipeline 换成桩。"""
    app.dependency_overrides[get_pipeline_dep] = lambda: FakePipeline()
    return TestClient(app)


def teardown_function() -> None:
    """每个用例结束后清掉 override，避免污染其他测试。"""
    app.dependency_overrides.clear()


def test_meta_and_documents() -> None:
    """/meta 返回枚举；/documents 默认身份（general/public）只能看到公开文档。"""
    client = _client()
    meta = client.get("/api/v1/meta").json()
    assert "ops" in meta["departments"]
    assert "confidential" in meta["sensitivities"]

    documents = client.get("/api/v1/documents").json()
    assert documents["documents"] == []


def test_documents_filtered_by_identity() -> None:
    """/documents 按身份过滤：ops/internal 看不到 rd/confidential，rd/confidential 全能看到。"""
    client = _client()

    ops = client.get(
        "/api/v1/documents", params={"department": "ops", "clearance": "internal"}
    ).json()
    assert [doc["rel_path"] for doc in ops["documents"]] == ["ops/internal/upi-zx-bank-asia.pdf"]

    # rd/confidential 只能看 rd 和 general 部门，密级再高也看不到 ops 的文档
    rd = client.get(
        "/api/v1/documents", params={"department": "rd", "clearance": "confidential"}
    ).json()
    assert [doc["rel_path"] for doc in rd["documents"]] == ["rd/confidential/fraud-transaction.md"]

    # 身份非法时和其他端点一样返回 400，而不是 500
    bad = client.get("/api/v1/documents", params={"department": "unknown"})
    assert bad.status_code == 400


def test_search_and_chat() -> None:
    """/search 返回上下文和调试信息；/chat 返回答案、引用和调试信息。"""
    client = _client()
    payload = {
        "query": "UPI耗尽了怎么处理？",
        "department": "ops",
        "clearance": "internal",
    }
    search = client.post("/api/v1/search", json=payload)
    assert search.status_code == 200
    assert search.json()["contexts"][0]["department"] == "ops"

    chat = client.post("/api/v1/chat", json={**payload, "session_id": "test", "debug": True})
    assert chat.status_code == 200
    body = chat.json()
    assert "10 万卢比" in body["answer"]
    assert body["citations"][0]["source_path"].startswith("ops/")
    assert body["debug"]["retrieval"]["dense_count"] == 1


def test_invalid_department_returns_400() -> None:
    """未知部门应返回 400 而不是 500。"""
    client = _client()
    response = client.post(
        "/api/v1/search",
        json={"query": "测试", "department": "unknown", "clearance": "public"},
    )
    assert response.status_code == 400


def test_empty_query_returns_422() -> None:
    """空问题会被 Pydantic 的 min_length 拦成 422。"""
    client = _client()
    response = client.post(
        "/api/v1/search",
        json={"query": "", "department": "ops", "clearance": "internal"},
    )
    assert response.status_code == 422


def test_openapi_contains_endpoints() -> None:
    """OpenAPI 文档必须包含所有对外端点。"""
    client = _client()
    spec = client.get("/openapi.json").json()
    for path in ["/api/v1/chat", "/api/v1/search", "/api/v1/health", "/api/v1/documents"]:
        assert path in spec["paths"]
