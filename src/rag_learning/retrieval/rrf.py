"""Reciprocal Rank Fusion：把两路有序结果融合成一个分数。"""

from __future__ import annotations


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """对多路排名做 RRF 融合，返回按分数降序的 (key, score) 列表。

    公式：score(key) = Σ 1 / (k + rank)，只依赖名次、不依赖原始分数，
    因此 dense（内积）和 sparse（词权重）这两种量纲不同的分数可以直接合并。
    k 越大，排名靠后的结果权重越接近，默认 60 是常用取值。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
