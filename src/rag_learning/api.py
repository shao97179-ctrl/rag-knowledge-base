"""FastAPI 接口：检索、问答、文档列表、健康检查；同时托管单页 Web UI。"""

from __future__ import annotations

import logging
from dataclasses import asdict
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
    description="BGE-M3 混合检索 + 权限隔离 + 可量化评估。身份切换仅用于演示，不是生产鉴权。",
)
# 演示用途放开所有来源；生产应改成白名单
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会话历史存在进程内存里，重启即丢；生产应换成 Redis/DB
_SESSIONS: dict[str, list[dict[str, str]]] = {}
_SESSION_TURNS = 10


class SearchRequest(BaseModel):
    """只检索的请求体。"""

    query: str = Field(..., min_length=1, max_length=500, description="用户问题")
    department: str = Field("general", description="部门：general/hr/finance/ops/rd")
    clearance: str = Field("public", description="密级：public/internal/confidential")
    top_k: int | None = Field(None, ge=1, le=20, description="最终父块数量")


class ChatRequest(SearchRequest):
    """问答请求体，在检索参数之上增加会话和调试开关。"""

    session_id: str | None = Field(None, max_length=64, description="多轮会话 ID")
    debug: bool = Field(False, description="是否返回检索调试信息")


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


def _remember(session_id: str, question: str, answer: str) -> None:
    """把一轮问答追加到内存会话，只保留最近 _SESSION_TURNS 轮。"""
    history = _SESSIONS.setdefault(session_id, [])
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    if len(history) > _SESSION_TURNS * 2:
        del history[: len(history) - _SESSION_TURNS * 2]


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
def documents(pipeline: RAGPipeline = Depends(get_pipeline_dep)) -> dict:
    """已入库文档列表，含权限和块数。"""
    return {"documents": [asdict(record) for record in pipeline.registry.list_all()]}


@app.post("/api/v1/search")
def search(req: SearchRequest, pipeline: RAGPipeline = Depends(get_pipeline_dep)) -> dict:
    """只检索不生成，返回上下文和双路召回调试信息。"""
    user = _user(req.department, req.clearance)
    result = pipeline.search(req.query, user, req.top_k)
    return {
        "query": req.query,
        "contexts": [_chunk_dict(chunk) for chunk in result.contexts],
        "debug": result.debug,
    }


@app.post("/api/v1/chat")
def chat(req: ChatRequest, pipeline: RAGPipeline = Depends(get_pipeline_dep)) -> dict:
    """检索 + 生成，可选多轮会话和调试信息。"""
    user = _user(req.department, req.clearance)
    history = _SESSIONS.get(req.session_id) if req.session_id else None
    result = pipeline.ask(req.query, user, history, req.top_k)
    if req.session_id:
        # 只记录本轮原始问题和最终答案，历史里不存改写后的问题
        _remember(req.session_id, req.query, result.answer)
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
