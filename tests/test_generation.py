import asyncio
from types import SimpleNamespace

from rag_learning.generation import (
    build_citations,
    build_user_prompt,
    generate_answer,
    is_refusal,
    stream_answer,
)
from rag_learning.schema import RetrievedChunk


class StreamingLLM:
    """可用的 LLM 桩：astream 分两段吐出答案。"""

    available = True

    async def achat(self, *args, **kwargs):
        """流式路径不会调用 achat，调到即失败。"""
        raise AssertionError("流式路径不应调用 achat")

    async def astream(self, system, user, model=None, temperature=None):
        """两段增量，引用编号在第二段。"""
        yield "住房贷款最长可贷 "
        yield "30 年[1]。"


def test_stream_answer_yields_deltas_then_result() -> None:
    """流式生成：先 delta 后 done，引用按答案里的编号过滤。"""
    events = []
    for event in asyncio.run(_collect(stream_answer(
        "住房贷款年限",
        [_context(1), _context(2)],
        StreamingLLM(),
        SimpleNamespace(max_context_chars=6000, llm_model="test-model"),
    ))):
        events.append(event)

    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert deltas == ["住房贷款最长可贷 ", "30 年[1]。"]
    done = events[-1]["result"]
    assert done.answer == "住房贷款最长可贷 30 年[1]。"
    assert [citation.index for citation in done.citations] == [1]
    assert done.refused is False


async def _collect(generator):
    """把异步生成器收成列表（给 asyncio.run 用）。"""
    return [event async for event in generator]


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
    result = asyncio.run(generate_answer(
        "怎么处理？",
        contexts,
        OfflineLLM(),
        SimpleNamespace(max_context_chars=6000),
    ))
    assert result.offline is True
    assert result.refused is False
    assert "[1]" in result.answer
    assert [citation.index for citation in result.citations] == [1, 2, 3]


def test_no_context_is_refusal() -> None:
    """检索为空时直接拒答，不调用 LLM。"""
    result = asyncio.run(generate_answer(
        "无关问题",
        [],
        OfflineLLM(),
        SimpleNamespace(max_context_chars=6000),
    ))
    assert result.refused is True
    assert is_refusal(result.answer)


def test_user_prompt_budget_counts_source_lines() -> None:
    """「来源：」前缀行算进字符预算，资料部分总长不能超过 max_chars。"""
    contexts = [_context(1), _context(2)]
    prompt = build_user_prompt("问题", contexts, max_chars=300)
    # 只统计 <资料> 标签内的部分
    body = prompt.split("<资料>\n", 1)[1].rsplit("\n</资料>", 1)[0]
    assert len(body) <= 300


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
