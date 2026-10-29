"""reranker 离线测试：只验证可用性判断和跳过逻辑，不加载模型。"""

from pathlib import Path

from rag_learning.reranker import BgeReranker


def test_missing_model_dir_is_unavailable(tmp_path: Path) -> None:
    """模型目录不存在时自动视为不可用，rerank 直接返回空列表。"""
    reranker = BgeReranker(model_path=tmp_path / "no-such-model", enabled=True)
    assert reranker.available is False
    assert reranker.rerank("查询", ["文档一", "文档二"]) == []
    assert reranker.loaded is False


def test_disabled_reranker_is_unavailable(tmp_path: Path) -> None:
    """显式关闭（RAG_RERANK_ENABLED=false）时即使目录存在也不启用。"""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    reranker = BgeReranker(model_path=model_dir, enabled=False)
    assert reranker.available is False
    assert reranker.rerank("查询", ["文档"]) == []
