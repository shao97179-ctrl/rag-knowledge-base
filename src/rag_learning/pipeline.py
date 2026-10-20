"""端到端管道：检索 + 改写 + 生成，API / CLI / 评估共用。"""

from __future__ import annotations

import logging
from functools import lru_cache

from .config import Settings, get_settings
from .embedder import get_embedder
from .generation import generate_answer
from .llm import get_llm
from .registry import Registry
from .retrieval import HybridRetriever, QueryRewriter, RetrievalResult
from .schema import AnswerResult, UserContext
from .vector_store import MilvusStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """把编码器、向量库、注册表、LLM 和检索器组装成一条完整链路。"""

    def __init__(self, settings: Settings | None = None) -> None:
        """所有组件都是进程内单例或轻量对象，构造函数本身不加载模型。"""
        self.settings = settings or get_settings()
        self.embedder = get_embedder(self.settings)
        self.store = MilvusStore(
            uri=self.settings.milvus_uri,
            collection=self.settings.milvus_collection,
            dimension=self.embedder.DIMENSION,
        )
        self.registry = Registry(self.settings.registry_path)
        self.llm = get_llm()
        self.rewriter = QueryRewriter(self.llm, self.settings.llm_light_model)
        self.retriever = HybridRetriever(self.settings, self.embedder, self.store)

    def search(self, query: str, user: UserContext, top_k: int | None = None) -> RetrievalResult:
        """只检索不生成，供 UI 的调试面板和 /search 接口使用。"""
        return self.retriever.search(query, user, top_k)

    def ask(
        self,
        query: str,
        user: UserContext,
        history: list[dict] | None = None,
        top_k: int | None = None,
    ) -> AnswerResult:
        """完整问答：多轮改写 -> 混合检索 -> 生成答案。"""
        rewritten = self.rewriter.rewrite(query, history)
        retrieved = self.retriever.search(rewritten, user, top_k)
        result = generate_answer(rewritten, retrieved.contexts, self.llm, self.settings)
        # 改写后的问题可能和原始问题不同，回填给调用方展示
        result.rewritten_query = rewritten
        result.debug.update({"retrieval": retrieved.debug})
        return result

    def health(self) -> dict:
        """汇总 LLM、编码器、向量库和文档数，供 /health 和 stats 命令使用。"""
        return {
            "llm": {
                "provider": self.settings.llm_provider,
                "model": self.settings.llm_model,
                "available": self.llm.available,
            },
            "embedder": {
                "path": str(self.settings.bge_m3_path),
                "loaded": self.embedder.loaded,
                "device": self.embedder.device,
                "dtype": str(self.embedder.dtype).replace("torch.", ""),
                "memory_mb": self.embedder.memory_mb,
            },
            "milvus": self.store.stats(),
            "documents": len(self.registry.list_all()),
        }

    def close(self) -> None:
        """关闭向量库连接；编码器留在单例里不主动释放。"""
        self.store.close()


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """进程内单例：API 多个请求共享同一个 pipeline。"""
    return RAGPipeline()
