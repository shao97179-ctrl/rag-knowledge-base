"""Reciprocal Rank Fusion：把两路有序结果融合成一个分数，支持按路加权。"""

from __future__ import annotations


def rrf_fuse(rankings: list[list[str]], k: int = 60, weights: list[float] | None = None) -> list[tuple[str, float]]:
    """对多路排名做加权 RRF 融合，返回按分数降序的 (key, score) 列表。

    公式：score(key) = Σ w_i / (k + rank_i)，只依赖名次、不依赖原始分数，
    因此 dense（内积）和 sparse（词权重）这两种量纲不同的分数可以直接合并。
    k 越大，排名靠后的结果权重越接近，默认 60 是常用取值。
    weights 为 None 时各路等权，等价于经典 RRF；dense 语义强就调高
    RAG_RRF_DENSE_WEIGHT，专有名词场景就调高 RAG_RRF_SPARSE_WEIGHT。
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
