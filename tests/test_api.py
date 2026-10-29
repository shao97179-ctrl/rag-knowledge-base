from types import SimpleNamespace
import json

from fastapi.testclient import TestClient

from rag_learning import api
from rag_learning.api import app, get_pipeline_dep
from rag_learning.registry import DocumentRecord
from rag_learning.retrieval.hybrid import RetrievalResult
from rag_learning.schema import AnswerResult, Citation, RetrievedChunk, UserContext


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

    def stats(self, expression=None):
        """返回假的集合状态；带 expression 时模拟按 ACL 过滤后的行数。"""
        rows = 2 if expression and "rd" in expression else 1
        return {"rows": rows, "collection": "test", "exists": True}


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

    async def ask(self, query, user, history=None, top_k=None):
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

    async def ask_stream(self, query, user, history=None, top_k=None):
        """模拟流式问答：两个 delta 之后跟完整结果。"""
        yield {"type": "delta", "text": "UPI 单笔限额为 "}
        yield {"type": "delta", "text": "10 万卢比 [1]。"}
        result = await self.ask(query, user, history, top_k)
        yield {"type": "done", "result": result}


def _client() -> TestClient:
    """用 dependency_overrides 把真实 pipeline 换成桩。"""
    app.dependency_overrides[get_pipeline_dep] = lambda: FakePipeline()
    return TestClient(app)


def teardown_function() -> None:
    """每个用例结束后清掉 override 和会话，避免污染其他测试。"""
    app.dependency_overrides.clear()
    api._SESSIONS.clear()


def test_meta_and_documents() -> None:
    """/meta 返回枚举且计数按身份过滤；/documents 默认身份只能看到公开文档。"""
    client = _client()
    meta = client.get("/api/v1/meta").json()
    assert "ops" in meta["departments"]
    assert "confidential" in meta["sensitivities"]
    # 默认 general/public：两条记录（ops/internal、rd/confidential）都不可见
    assert meta["documents"] == 0
    assert meta["milvus"]["rows"] == 1

    # rd/confidential 身份：文档数和过滤后的行数都该跟着变
    rd_meta = client.get("/api/v1/meta", params={"department": "rd", "clearance": "confidential"}).json()
    assert rd_meta["documents"] == 1
    assert rd_meta["milvus"]["rows"] == 2

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


def test_search_debug_is_opt_in() -> None:
    """/search 默认不返回 debug，显式传 debug=true 才返回。"""
    client = _client()
    payload = {"query": "UPI 限额", "department": "ops", "clearance": "internal"}

    default = client.post("/api/v1/search", json=payload).json()
    assert default["debug"] == {}

    opened = client.post("/api/v1/search", json={**payload, "debug": True}).json()
    assert opened["debug"]["filter_expr"]


def test_session_history_capped_and_identity_bound() -> None:
    """会话轮数有上限；同一个 session_id 换身份是另一个会话，历史不共享。"""
    ops = UserContext("ops", "internal")
    finance = UserContext("finance", "internal")
    key = api._session_key("s1", ops)

    for index in range(15):
        api._remember(key, f"问{index}", f"答{index}")

    # 上限是 10 轮 = 20 条，15 轮写进去只留最近 10 轮
    history = api._get_history(key)
    assert len(history) == api._SESSION_TURNS * 2
    assert history[0] == {"role": "user", "content": "问5"}
    assert api._get_history(api._session_key("s1", finance)) is None


def test_session_ttl_and_total_limit(monkeypatch) -> None:
    """闲置过期的会话会被清掉；总量到上限后最久未用的先淘汰。"""
    # TTL 设成 -1：刚写入的会话也立刻判定过期
    monkeypatch.setattr(api, "_SESSION_TTL", -1)
    api._remember("k1", "问", "答")
    assert api._get_history("k1") is None
    assert "k1" not in api._SESSIONS

    monkeypatch.setattr(api, "_SESSION_TTL", 60)
    monkeypatch.setattr(api, "_SESSION_LIMIT", 1)
    api._remember("k1", "问", "答")
    api._remember("k2", "问", "答")
    assert "k1" not in api._SESSIONS
    assert "k2" in api._SESSIONS


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


def test_chat_stream_sse() -> None:
    """/chat/stream 返回 SSE：delta 事件逐段，done 事件带引用。"""
    client = _client()
    response = client.post(
        "/api/v1/chat/stream",
        json={"query": "UPI 限额", "department": "ops", "clearance": "internal"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = [
        json.loads(line[len("data: "):])
        for line in response.text.split("\n\n")
        if line.startswith("data: ")
    ]
    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert deltas == ["UPI 单笔限额为 ", "10 万卢比 [1]。"]
    done = events[-1]
    assert done["type"] == "done"
    assert done["citations"][0]["source_path"].startswith("ops/")
    assert "10 万卢比" in done["answer"]


def test_rate_limit_returns_429(monkeypatch) -> None:
    """每 IP 每分钟超出限额后返回 429。"""
    monkeypatch.setattr(api, "get_settings", lambda: SimpleNamespace(rate_limit_per_minute=2))
    api._RATE_BUCKETS.clear()
    client = _client()
    payload = {"query": "UPI", "department": "ops", "clearance": "internal"}
    assert client.post("/api/v1/search", json=payload).status_code == 200
    assert client.post("/api/v1/search", json=payload).status_code == 200
    assert client.post("/api/v1/search", json=payload).status_code == 429


def test_request_id_header() -> None:
    """每个响应都带 x-request-id，透传上游传入的值。"""
    client = _client()
    auto = client.get("/api/v1/meta")
    assert auto.headers.get("x-request-id")
    passed = client.get("/api/v1/meta", headers={"x-request-id": "my-trace-42"})
    assert passed.headers["x-request-id"] == "my-trace-42"


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
