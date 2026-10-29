from rag_learning.retrieval.filters import build_acl_expr
from rag_learning.retrieval.rrf import rrf_fuse
from rag_learning.schema import UserContext


def test_rrf_fuse_known_ranking() -> None:
    """RRF 只依赖名次：两路都第一的 a 和 b 分数相同，且高于只在单路出现的项。"""
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [key for key, _ in fused] == ["a", "b", "c", "d"]
    scores = dict(fused)
    # a 在 dense 第 1、sparse 第 2；b 在 dense 第 2、sparse 第 1，总和相同
    assert scores["a"] == scores["b"]
    assert scores["a"] > scores["c"]
    assert scores["c"] == scores["d"]


def test_rrf_fuse_weights_flip_ranking() -> None:
    """加权 RRF：给第二路足够大的权重可以让它的第一名反超。"""
    fused = rrf_fuse([["a", "b"], ["b", "a"]], k=60, weights=[1.0, 3.0])
    assert fused[0][0] == "b"
    # 等权时应该回到经典 RRF 的并列，说明权重真的参与了计算
    equal = rrf_fuse([["a", "b"], ["b", "a"]], k=60, weights=[1.0, 1.0])
    assert dict(equal)["a"] == dict(equal)["b"]


def test_acl_expression() -> None:
    """ACL 表达式要包含用户部门 + general，以及所有不高于自身密级的级别。"""
    assert build_acl_expr(UserContext("ops", "internal")) == (
        'department in ["ops", "general"] and sensitivity in ["public", "internal"]'
    )
    # general 部门不需要重复追加
    assert build_acl_expr(UserContext("general", "confidential")) == (
        'department in ["general"] and sensitivity in ["public", "internal", "confidential"]'
    )
    assert build_acl_expr(UserContext("hr", "public")) == (
        'department in ["hr", "general"] and sensitivity in ["public"]'
    )
