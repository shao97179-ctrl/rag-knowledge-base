"""生成与降级：LLM 可用时生成带引用的答案，否则返回抽取式摘要。"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..llm import LLMClient
from ..schema import AnswerResult, Citation, RetrievedChunk
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)

# 答案里的引用标记，例如 "[1]"、"[12]"
CITATION_RE = re.compile(r"\[(\d+)\]")

# 拒答的判定话术。prompt 要求模型拒答时把结论放在答案开头，
# 所以只检查第一句；正文后半段提到这些词的（先答一半再补充说明）不算。
# 「资料中没有」单用太宽——「资料中没有写明 X，但 Y」是正常回答——收紧成「资料中没有相关」。
REFUSAL_PATTERNS = (
    "根据现有资料无法回答",
    "无法回答",
    "没有相关信息",
    "未找到相关",
    "不足以回答",
    "无法确定",
    "资料中没有相关",
)

# 转折词：话术后面紧跟转折、又给出实际内容的是「部分拒答 + 补充」，不当拒答处理
_TURN_WORDS = ("但", "不过", "然而", "可是")


def _first_sentence(text: str) -> str:
    """截取第一个句号 / 问号 / 感叹号 / 换行之前的内容。"""
    for index, char in enumerate(text):
        if char in "。！？!?\n":
            return text[:index]
    return text


def is_refusal(answer: str) -> bool:
    """判断答案是否拒答：话术要出现在第一句里，且后面没有转折补充。"""
    first = _first_sentence(answer.strip())
    if not first:
        return False
    for pattern in REFUSAL_PATTERNS:
        position = first.find(pattern)
        if position == -1:
            continue
        rest = first[position + len(pattern):]
        if any(word in rest for word in _TURN_WORDS):
            continue
        return True
    return False


def build_citations(
    contexts: list[RetrievedChunk],
    answer: str | None = None,
) -> list[Citation]:
    """只返回答案里真正引用过的编号；没有显式引用时返回全部。"""
    cited: set[int] | None = None
    if answer:
        # 提取 [n] 编号，并过滤掉超出上下文范围的越界编号
        cited = {int(match) for match in CITATION_RE.findall(answer)}
        cited = {index for index in cited if 1 <= index <= len(contexts)} or None

    citations: list[Citation] = []
    for index, context in enumerate(contexts, start=1):
        # cited 为 None 说明模型没写引用，此时退回「全部上下文都算来源」
        if cited is not None and index not in cited:
            continue
        # 引用卡片展示子块（更精准），没有子块时用父块
        snippet = (context.child_text or context.parent_text).strip().replace("\n", " ")
        citations.append(
            Citation(
                index=index,
                title=context.title,
                source_path=context.source_path,
                heading=context.heading,
                department=context.department,
                sensitivity=context.sensitivity,
                snippet=snippet[:160],
                score=round(context.score, 6),
            )
        )
    return citations


def _extractive_answer(contexts: list[RetrievedChunk], reason: str) -> str:
    """离线降级：直接把最相关的 3 个父块截断拼成答案，不做任何推理。"""
    parts = [f"（{reason}）根据检索结果，最相关的内容如下："]
    for index, context in enumerate(contexts[:3], start=1):
        text = (context.parent_text or context.child_text).strip().replace("\n", " ")
        parts.append(f"[{index}] {text[:280]}")
    return "\n\n".join(parts)


def generate_answer(
    query: str,
    contexts: list[RetrievedChunk],
    llm: LLMClient,
    settings: Any,
) -> AnswerResult:
    """基于检索上下文生成答案；无上下文直接拒答，LLM 失败则降级为抽取式。"""
    if not contexts:
        # 检索为空时不必调用 LLM，直接走拒答，既省钱又避免模型瞎编
        return AnswerResult(
            answer="根据现有资料无法回答。",
            citations=[],
            contexts=[],
            rewritten_query=query,
            refused=True,
            offline=not llm.available,
            debug={"reason": "no_context"},
        )

    offline = not llm.available
    answer: str | None = None
    if llm.available:
        answer = llm.chat(
            SYSTEM_PROMPT,
            build_user_prompt(query, contexts, settings.max_context_chars),
            model=settings.llm_model,
        )

    if answer is None:
        # LLM 不可用或调用失败：降级为抽取式，并标记 offline 让前端提示
        reason = "离线模式" if offline else "LLM 调用失败，已降级"
        if not offline:
            logger.warning("生成失败，降级为抽取式回答")
        answer = _extractive_answer(contexts, reason)
        offline = True

    refused = is_refusal(answer)
    # 拒答时不给引用，避免把无关资料当成「来源」展示
    citations = [] if refused else build_citations(contexts, answer)
    return AnswerResult(
        answer=answer,
        citations=citations,
        contexts=contexts,
        rewritten_query=query,
        refused=refused,
        offline=offline,
        debug={"reason": "refusal" if refused else "answered"},
    )
