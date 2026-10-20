"""混合检索：dense + sparse 双路召回，RRF 融合，父块去重。"""

from __future__ import annotations

import logging
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
    """双路召回 + RRF 融合 + 父块去重。"""

    def __init__(self, settings: Any, embedder: BgeM3Embedder, store: MilvusStore) -> None:
        """settings 提供 top_k / rrf_k 等参数，embedder 负责查询编码。"""
        self.settings = settings
        self.embedder = embedder
        self.store = store

    def search(
        self,
        query: str,
        user: UserContext,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """执行一次混合检索，返回去重后的父块上下文。

        流程：清洗 query -> 双路向量 -> 两路检索（带 ACL）-> RRF 融合
        -> 按 parent_id 去重 -> 生成前再做一次权限校验。
        """
        query = clean_query(query)
        top_k = top_k or self.settings.top_k_final
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
        )

        contexts: list[RetrievedChunk] = []
        seen_parents: set[str] = set()
        for pk, score in fused:
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
            "检索完成 query=%r user=%s/%s filter=%s dense=%s sparse=%s fused=%s parents=%s",
            query,
            user.department,
            user.clearance,
            expression,
            len(dense_hits),
            len(sparse_hits),
            len(fused),
            len(contexts),
        )
        return RetrievalResult(
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
                # 只回传前 10 条，避免 debug 响应过大
                "dense_hits": [self._brief(hit, "distance") for hit in dense_hits[:10]],
                "sparse_hits": [self._brief(hit, "distance") for hit in sparse_hits[:10]],
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
