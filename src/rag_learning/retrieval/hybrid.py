"""混合检索：dense + sparse 双路召回，RRF 融合，可选 cross-encoder 精排，父块去重。"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from ..cleaner import clean_query
from ..embedder import BgeM3Embedder
from ..schema import RetrievedChunk, UserContext
from ..vector_store import MilvusStore
from .filters import build_acl_expr
from .rrf import rrf_fuse

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果 + 供 UI 展示的调试信息。"""

    contexts: list[RetrievedChunk] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """双路召回 + RRF 融合 + 可选精排 + 父块去重 + 结果缓存。"""

    def __init__(
        self,
        settings: Any,
        embedder: BgeM3Embedder,
        store: MilvusStore,
        reranker: Any | None = None,
    ) -> None:
        """settings 提供 top_k / rrf_k 等参数，embedder 负责查询编码。

        reranker 传 None 或模型不可用时跳过精排，检索保持纯 RRF 排序。
        """
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        # LRU 缓存：key 含索引行数，入库/删文档后行数变化，旧缓存自然失效
        self._cache: OrderedDict[tuple, list[RetrievedChunk]] = OrderedDict()
        self._cache_lock = threading.Lock()
        # 最近一次精排的调试信息，search 组装 debug 时取走
        self._last_rerank_debug: list[dict] | None = None

    def _cache_key(self, query: str, user: UserContext, top_k: int) -> tuple | None:
        """缓存 size 为 0 表示关闭；索引行数进 key，保证入库后自动失效。"""
        size = int(getattr(self.settings, "retrieval_cache_size", 0) or 0)
        if size <= 0:
            return None
        return (query, user.department, user.clearance, top_k, self.store.count())

    def _cache_get(self, key: tuple) -> list[RetrievedChunk] | None:
        """读缓存；命中会移到队头（LRU 语义）。"""
        with self._cache_lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return list(self._cache[key])

    def _cache_put(self, key: tuple, contexts: list[RetrievedChunk]) -> None:
        """写缓存并淘汰最久未用的条目。"""
        size = int(getattr(self.settings, "retrieval_cache_size", 0) or 0)
        with self._cache_lock:
            self._cache[key] = list(contexts)
            self._cache.move_to_end(key)
            while len(self._cache) > size:
                self._cache.popitem(last=False)

    def search(
        self,
        query: str,
        user: UserContext,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """执行一次混合检索，返回去重后的父块上下文。

        流程：清洗 query -> 双路向量 -> 两路检索（带 ACL）-> 加权 RRF 融合
        -> （可选）cross-encoder 精排 -> 按 parent_id 去重 -> 生成前再做一次权限校验。
        """
        query = clean_query(query)
        top_k = top_k or self.settings.top_k_final
        cache_key = self._cache_key(query, user, top_k)
        if cache_key is not None:
            cached = self._cache_get(cache_key)
            if cached is not None:
                logger.info("检索缓存命中 query=%r user=%s/%s", query, user.department, user.clearance)
                return RetrievalResult(contexts=cached, debug={"cached": True})

        dense_vector, sparse_vector = self.embedder.encode_query(query)
        # 两路用同一个 filter 表达式，保证候选集权限一致
        expression = build_acl_expr(user)

        dense_hits = self.store.search_dense(
            dense_vector, expression, self.settings.top_k_dense
        )
        sparse_hits = self.store.search_sparse(
            sparse_vector, expression, self.settings.top_k_sparse
        )

        # 按 pk 合并两路原始命中，同时保留各自的排名和分数用于展示
        by_pk: dict[str, dict] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            record = by_pk.setdefault(hit["pk"], dict(hit))
            record["dense_rank"] = rank
            record["dense_score"] = hit["distance"]
        for rank, hit in enumerate(sparse_hits, start=1):
            record = by_pk.setdefault(hit["pk"], dict(hit))
            record["sparse_rank"] = rank
            record["sparse_score"] = hit["distance"]

        fused = rrf_fuse(
            [
                [hit["pk"] for hit in dense_hits],
                [hit["pk"] for hit in sparse_hits],
            ],
            k=self.settings.rrf_k,
            weights=[
                float(getattr(self.settings, "rrf_dense_weight", 1.0) or 1.0),
                float(getattr(self.settings, "rrf_sparse_weight", 1.0) or 1.0),
            ],
        )

        candidates = self._rerank(query, fused, by_pk)

        contexts: list[RetrievedChunk] = []
        seen_parents: set[str] = set()
        for pk, score in candidates:
            record = by_pk[pk]
            # 防御性二次校验：即使 filter 写错，也不把无权内容交给生成层
            if not user.can_access(record["department"], record["sensitivity"]):
                logger.error(
                    "ACL 泄漏拦截：%s %s/%s 被用户 %s/%s 命中",
                    record["source_path"],
                    record["department"],
                    record["sensitivity"],
                    user.department,
                    user.clearance,
                )
                continue
            # 同一父块下的多个子块只保留排名最高的那个，节省上下文预算
            parent_id = record["parent_id"]
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            contexts.append(
                RetrievedChunk(
                    pk=record["pk"],
                    doc_id=record["doc_id"],
                    chunk_index=int(record["chunk_index"]),
                    parent_id=parent_id,
                    child_text=record["child_text"],
                    # parent_text 冗余存在行里，正常不会缺失，兜底用 child_text
                    parent_text=record.get("parent_text") or record["child_text"],
                    heading=record.get("heading", ""),
                    title=record.get("title", ""),
                    source_path=record["source_path"],
                    department=record["department"],
                    sensitivity=record["sensitivity"],
                    doc_format=record.get("doc_format", ""),
                    score=score,
                    dense_rank=record.get("dense_rank"),
                    sparse_rank=record.get("sparse_rank"),
                    dense_score=record.get("dense_score"),
                    sparse_score=record.get("sparse_score"),
                )
            )
            if len(contexts) >= top_k:
                break

        logger.info(
            "检索完成 query=%r user=%s/%s filter=%s dense=%s sparse=%s fused=%s parents=%s rerank=%s",
            query,
            user.department,
            user.clearance,
            expression,
            len(dense_hits),
            len(sparse_hits),
            len(fused),
            len(contexts),
            self.reranker is not None and self.reranker.available,
        )
        result = RetrievalResult(
            contexts=contexts,
            debug={
                "query": query,
                "department": user.department,
                "clearance": user.clearance,
                "filter_expr": expression,
                "dense_count": len(dense_hits),
                "sparse_count": len(sparse_hits),
                "fused_count": len(fused),
                "parent_count": len(contexts),
                # 全量回传：评估要拿原始召回做 ACL 审计，截断会让审计口径小于召回口径
                "dense_hits": [self._brief(hit, "distance") for hit in dense_hits],
                "sparse_hits": [self._brief(hit, "distance") for hit in sparse_hits],
                "fused": [
                    {
                        "pk": pk,
                        "doc_id": by_pk[pk]["doc_id"],
                        "source_path": by_pk[pk]["source_path"],
                        "rrf": round(score, 6),
                    }
                    for pk, score in fused[:10]
                ],
            },
        )
        if self._last_rerank_debug is not None:
            result.debug["rerank"] = self._last_rerank_debug
        if cache_key is not None:
            self._cache_put(cache_key, contexts)
        return result

    def _rerank(
        self,
        query: str,
        fused: list[tuple[str, float]],
        by_pk: dict[str, dict],
    ) -> list[tuple[str, float]]:
        """用 cross-encoder 对融合头部候选重排；不可用或失败时原样返回。

        只精排前 rerank_candidates 条：rerank 的收益集中在头部，
        长尾让 RRF 名次兜底，省下的推理时间比损失的质量多。
        """
        self._last_rerank_debug = None
        if self.reranker is None or not self.reranker.available:
            return fused
        limit = int(getattr(self.settings, "rerank_candidates", 20) or 20)
        head = fused[:limit]
        tail = fused[limit:]
        pks = [pk for pk, _ in head]
        try:
            scores = self.reranker.rerank(query, [by_pk[pk]["child_text"] for pk in pks])
        except Exception:  # noqa: BLE001
            logger.warning("rerank 失败，退回 RRF 排序", exc_info=True)
            return fused
        resorted = sorted(zip(pks, scores), key=lambda item: item[1], reverse=True)
        self._last_rerank_debug = [
            {
                "pk": pk,
                "doc_id": by_pk[pk]["doc_id"],
                "source_path": by_pk[pk]["source_path"],
                "rerank": round(score, 6),
            }
            for pk, score in resorted[:10]
        ]
        return [(pk, score) for pk, score in resorted] + tail

    @staticmethod
    def _brief(hit: dict, score_field: str) -> dict:
        """从完整 hit 里挑出 UI 需要的字段，控制调试信息的体积。"""
        return {
            "pk": hit["pk"],
            "doc_id": hit["doc_id"],
            "source_path": hit["source_path"],
            "title": hit.get("title", ""),
            "heading": hit.get("heading", ""),
            "department": hit["department"],
            "sensitivity": hit["sensitivity"],
            "score": round(float(hit.get(score_field, 0.0)), 4),
        }
