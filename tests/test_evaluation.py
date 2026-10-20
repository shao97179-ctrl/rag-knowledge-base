from rag_learning.evaluation.metrics import (
    find_acl_violations,
    judge_faithfulness,
    mrr_at_k,
    parse_score,
    recall_at_k,
)
from rag_learning.generation import is_refusal
from rag_learning.schema import RetrievedChunk, UserContext


def test_recall_and_mrr() -> None:
    """Recall@K / MRR@K 的边界：部分命中、完全不命中、K 截断。"""
    assert recall_at_k(["a", "b", "c"], ["b"], 5) == 1.0
    assert recall_at_k(["a", "b"], ["c", "d"], 5) == 0.0
    # K=2 时只看到 a、b，预期 b、c 只命中 1 个 -> 0.5
    assert recall_at_k(["a", "b", "c"], ["b", "c"], 2) == 0.5
    assert mrr_at_k(["a", "b", "c"], ["c"], 5) == 1 / 3
    assert mrr_at_k(["a"], ["z"], 5) == 0.0


def test_parse_score_formats() -> None:
    """裁判输出格式很野：标准 JSON、代码块、裸文本、智谱 XML 都要能解析。"""
    assert parse_score('{"score": 0.8, "unsupported": []}') == 0.8
    assert parse_score('```json\n{"score": 0.5}\n```') == 0.5
    assert parse_score('裁判结论： "score": 0.75 完毕') == 0.75
    assert parse_score("<arg_key>score</arg_key><arg_value>0.9</arg_value>") == 0.9
    assert parse_score("no score here") is None
    # 越界分数要夹到 [0, 1]
    assert parse_score('{"score": 1.5}') == 1.0


def test_refusal_counts_as_faithful() -> None:
    """正确拒答视为忠于资料，直接记 1.0，且不应调用裁判模型。"""
    class FakeLLM:
        available = True

        def chat(self, *args, **kwargs):
            """拒答路径不该走到裁判，调到就说明逻辑有 bug。"""
            raise AssertionError("拒答不应调用裁判")

    assert is_refusal("根据现有资料无法回答。")
    assert judge_faithfulness("问题", "根据现有资料无法回答。", [], FakeLLM(), "model") == 1.0


def _chunk(department: str, sensitivity: str) -> RetrievedChunk:
    """构造一条带权限字段的检索结果。"""
    return RetrievedChunk(
        pk="pk",
        doc_id="doc",
        chunk_index=0,
        parent_id="parent",
        child_text="text",
        parent_text="text",
        heading="",
        title="",
        source_path=f"{department}/{sensitivity}/x.md",
        department=department,
        sensitivity=sensitivity,
        doc_format="markdown",
    )


def test_acl_violations() -> None:
    """越权检测：ops/internal 看到 hr/confidential 必须被识别出来。"""
    user = UserContext("ops", "internal")
    assert find_acl_violations([_chunk("ops", "internal"), _chunk("general", "public")], user) == []
    violations = find_acl_violations([_chunk("hr", "confidential")], user)
    assert len(violations) == 1
