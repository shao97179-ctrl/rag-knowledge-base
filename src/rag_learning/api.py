"""FastAPI 接口：检索、问答、文档列表、健康检查；同时托管单页 Web UI。"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import RAGPipeline, get_pipeline
from .schema import DEPARTMENTS, SENSITIVITY_LEVELS, UserContext

logger = logging.getLogger(__name__)

app = FastAPI(
    title="企业知识库 RAG",
    version="0.1.0",
    description="BGE-M3 混合检索 + 权限隔离 + 可量化评估。",
)
# 演示用途放开所有来源；生产应改成白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会话历史存在进程内存里，重启即丢；生产应换成 Redis/DB
@dataclass
class _Session:
    """一个会话的改写历史和最近活跃时间。"""

    history: list[dict[str, str]] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)


_SESSIONS: dict[str, _Session] = {}
_SESSION_TURNS = 10
# 会话总量和闲置过期都设上限，长驻进程的内存不会随访问量无限涨
_SESSION_LIMIT = 1000
_SESSION_TTL = 24 * 60 * 60


class SearchRequest(BaseModel):
    """只检索的请求体。"""

    query: str = Field(..., min_length=1, max_length=500, description="用户问题")
    department: str = Field("general", description="部门：general/hr/finance/ops/rd")
    clearance: str = Field("public", description="密级：public/internal/confidential")
    top_k: int | None = Field(None, ge=1, le=20, description="最终父块数量")
    debug: bool = Field(False, description="是否返回检索调试信息")


class ChatRequest(SearchRequest):
    """问答请求体，在检索参数之上增加会话 ID（debug 继承自检索请求）。"""

    session_id: str | None = Field(None, max_length=64, description="多轮会话 ID")


def get_pipeline_dep() -> RAGPipeline:
    """FastAPI 依赖，单独抽成函数是为了测试里能 override。"""
    return get_pipeline()


def _user(department: str, clearance: str) -> UserContext:
    """把请求里的部门/密级转成 UserContext，非法值返回 400。"""
    try:
        return UserContext(department, clearance)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _chunk_dict(chunk) -> dict[str, Any]:
    """RetrievedChunk 转 JSON 友好的 dict，浮点数做定点截断。"""
    return {
        "pk": chunk.pk,
        "doc_id": chunk.doc_id,
        "source_path": chunk.source_path,
        "title": chunk.title,
        "heading": chunk.heading,
        "department": chunk.department,
        "sensitivity": chunk.sensitivity,
        "doc_format": chunk.doc_format,
        "score": round(chunk.score, 6),
        "dense_rank": chunk.dense_rank,
        "sparse_rank": chunk.sparse_rank,
        "dense_score": None if chunk.dense_score is None else round(chunk.dense_score, 4),
        "sparse_score": None if chunk.sparse_score is None else round(chunk.sparse_score, 4),
        "child_text": chunk.child_text,
        "parent_text": chunk.parent_text,
    }


def _session_key(session_id: str, user: UserContext) -> str:
    """会话槽位绑定「身份 + 会话 ID」：换部门/密级等于开新会话，
    一份身份的改写历史不会喂给另一个身份的提问。"""
    return f"{user.department}/{user.clearance}/{session_id}"


def _get_history(key: str) -> list[dict[str, str]] | None:
    """取会话历史；闲置过期的会话就地清掉并返回 None。"""
    session = _SESSIONS.get(key)
    if session is None:
        return None
    if time.monotonic() - session.last_used > _SESSION_TTL:
        del _SESSIONS[key]
        return None
    return session.history


def _remember(key: str, question: str, answer: str) -> None:
    """追加一轮问答，只保留最近 _SESSION_TURNS 轮。

    写入前先清过期会话；总量到上限时淘汰最久未用的。
    """
    now = time.monotonic()
    expired = [k for k, session in _SESSIONS.items() if now - session.last_used > _SESSION_TTL]
    for k in expired:
        del _SESSIONS[k]
    if key not in _SESSIONS and len(_SESSIONS) >= _SESSION_LIMIT:
        oldest = min(_SESSIONS, key=lambda k: _SESSIONS[k].last_used)
        del _SESSIONS[oldest]
    session = _SESSIONS.setdefault(key, _Session())
    session.last_used = now
    session.history.append({"role": "user", "content": question})
    session.history.append({"role": "assistant", "content": answer})
    if len(session.history) > _SESSION_TURNS * 2:
        del session.history[: len(session.history) - _SESSION_TURNS * 2]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """返回单页 Web UI。"""
    path = get_settings().web_dir / "index.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Web UI 文件不存在")
    return FileResponse(path)


@app.get("/api/v1/health")
def health(pipeline: RAGPipeline = Depends(get_pipeline_dep)) -> dict:
    """健康检查：LLM、编码器、Milvus、文档数量。"""
    return pipeline.health()


@app.get("/api/v1/meta")
def meta(pipeline: RAGPipeline = Depends(get_pipeline_dep)) -> dict:
    """前端初始化所需的枚举和状态（部门、密级、LLM、索引）。"""
    return {
        "departments": list(DEPARTMENTS),
        "sensitivities": list(SENSITIVITY_LEVELS),
        "llm": {
            "provider": pipeline.settings.llm_provider,
            "model": pipeline.settings.llm_model,
            "available": pipeline.llm.available,
        },
        "milvus": pipeline.store.stats(),
        "documents": len(pipeline.registry.list_all()),
    }


@app.get("/api/v1/documents")
def documents(
    department: str = "general",
    clearance: str = "public",
    pipeline: RAGPipeline = Depends(get_pipeline_dep),
) -> dict:
    """按当前身份过滤后的文档列表；无权文档连元数据都不返回。"""
    user = _user(department, clearance)
    visible = [
        asdict(record)
        for record in pipeline.registry.list_all()
        if user.can_access(record.department, record.sensitivity)
    ]
    return {"documents": visible}


@app.post("/api/v1/search")
def search(req: SearchRequest, pipeline: RAGPipeline = Depends(get_pipeline_dep)) -> dict:
    """只检索不生成；debug=true 时才返回双路召回调试信息，口径与 /chat 一致。"""
    user = _user(req.department, req.clearance)
    result = pipeline.search(req.query, user, req.top_k)
    return {
        "query": req.query,
        "contexts": [_chunk_dict(chunk) for chunk in result.contexts],
        # debug 默认不回传，避免暴露内部检索细节
        "debug": result.debug if req.debug else {},
    }


@app.post("/api/v1/chat")
def chat(req: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline_dep)) -> dict:
    """检索 + 生成，可选多轮会话和调试信息。"""
    user = _user(req.department, req.clearance)
    key = _session_key(req.session_id, user) if req.session_id else None
    history = _get_history(key) if key else None
    result = pipeline.ask(req.query, user, history, req.top_k)
    if key:
        # 只记录本轮原始问题和最终答案，历史里不存改写后的问题
        _remember(key, req.query, result.answer)
    return {
        "answer": result.answer,
        "refused": result.refused,
        "offline": result.offline,
        "rewritten_query": result.rewritten_query,
        "citations": [asdict(citation) for citation in result.citations],
        "contexts": [_chunk_dict(chunk) for chunk in result.contexts],
        # debug 默认不回传，避免暴露内部检索细节
        "debug": result.debug if req.debug else {},
    }
