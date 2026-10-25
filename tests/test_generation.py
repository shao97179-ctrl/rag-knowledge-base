from types import SimpleNamespace

from rag_learning.generation import build_citations, generate_answer, is_refusal
from rag_learning.schema import RetrievedChunk


def _context(index: int) -> RetrievedChunk:
    """构造一条检索上下文，index 同时用作编号和分数。"""
    return RetrievedChunk(
        pk=f"pk{index}",
        doc_id=f"doc{index}",
        chunk_index=index,
        parent_id=f"parent{index}",
        child_text=f"第 {index} 段内容。",
        parent_text=f"第 {index} 段内容，包含更多上下文。",
        heading=f"章节 {index}",
        title=f"文档 {index}",
        source_path=f"ops/internal/doc{index}.md",
        department="ops",
        sensitivity="internal",
        doc_format="markdown",
        score=0.01 * index,
    )


def test_citations_only_include_cited_numbers() -> None:
    """答案里显式引用 [1][3] 时只返回这两条；没写引用时退回全部。"""
    contexts = [_context(1), _context(2), _context(3)]
    citations = build_citations(contexts, "先看 [1]，再看 [3]。")
    assert [citation.index for citation in citations] == [1, 3]

    all_citations = build_citations(contexts, "没有显式引用")
    assert [citation.index for citation in all_citations] == [1, 2, 3]


class OfflineLLM:
    """离线 LLM 桩：available=False，不实现 chat。"""

    available = False


def test_offline_generation_is_extractive() -> None:
    """离线模式返回抽取式答案，并给全部上下文加引用。"""
    contexts = [_context(1), _context(2), _context(3)]
    result = generate_answer(
        "怎么处理？",
        contexts,
        OfflineLLM(),
        SimpleNamespace(max_context_chars=6000),
    )
    assert result.offline is True
    assert result.refused is False
    assert "[1]" in result.answer
    assert [citation.index for citation in result.citations] == [1, 2, 3]


def test_no_context_is_refusal() -> None:
    """检索为空时直接拒答，不调用 LLM。"""
    result = generate_answer(
        "无关问题",
        [],
        OfflineLLM(),
        SimpleNamespace(max_context_chars=6000),
    )
    assert result.refused is True
    assert is_refusal(result.answer)


def test_hedged_answer_is_not_refusal() -> None:
    """先说资料没写什么、再给实际内容的答案不能判成拒答。"""
    # 话术在句子开头但紧跟转折，属于「部分拒答 + 补充」
    assert not is_refusal("资料中没有写明具体日期，但住房贷款最长可贷 30 年[1]。")
    assert not is_refusal("无法确定具体利率，不过资料写明年限最长 30 年。")
    # 话术出现在后半段，第一句已经是正常回答
    assert not is_refusal("住房贷款最长可贷 30 年[1]。资料中没有提及利率。")


def test_refusal_variants_are_still_caught() -> None:
    """拒答话术在第一句、后面跟建议或省略句号的都算拒答。"""
    assert is_refusal("根据现有资料无法回答。")
    assert is_refusal("根据现有资料无法回答。建议携带证件到网点咨询。")
    assert is_refusal("资料中没有相关信息")
    assert is_refusal("抱歉，这个问题未找到相关说明。")
