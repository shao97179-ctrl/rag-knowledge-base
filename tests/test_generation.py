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
