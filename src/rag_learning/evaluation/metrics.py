"""评估指标：Recall@K、MRR@K、Faithfulness（LLM 裁判）、ACL 违规检测。"""

from __future__ import annotations

import json
import logging
import re

from ..generation import is_refusal
from ..llm import LLMClient
from ..schema import RetrievedChunk, UserContext

logger = logging.getLogger(__name__)

FAITHFULNESS_SYSTEM = """你是严格的 RAG 评估裁判。给定问题、回答和资料，判断回答中的事实是否被资料支持。
只输出一个 JSON 对象，格式为：{"score": 0.0, "unsupported": ["不被支持的句子"]}
score 取值：1.0 表示全部有依据，0.5 表示部分有依据，0.0 表示完全没有依据。
不要输出 JSON 以外的任何内容。"""


def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int = 5) -> float:
    """Recall@K：前 K 个结果覆盖了多少比例的预期文档。

    没有预期文档时返回 1.0（视为无需召回），避免拉低整体均值。
    """
    if not expected_doc_ids:
        return 1.0
    top = set(retrieved_doc_ids[:k])
    hits = sum(1 for doc_id in expected_doc_ids if doc_id in top)
    return hits / len(expected_doc_ids)


def mrr_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int = 5) -> float:
    """MRR@K：第一个相关结果排名的倒数；K 内没有相关结果记 0。"""
    if not expected_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def find_acl_violations(chunks: list[RetrievedChunk], user: UserContext) -> list[RetrievedChunk]:
    """返回当前用户无权访问的 chunk 列表，正常情况下应为空。"""
    return [chunk for chunk in chunks if not user.can_access(chunk.department, chunk.sensitivity)]


def parse_score(raw: str | None) -> float | None:
    """兼容 JSON、代码块、以及智谱有时返回的 arg_key/arg_value XML。"""
    if not raw:
        return None
    text = raw.strip().strip("`")
    # 第一优先：截取第一个 { 到最后一个 } 按 JSON 解析，能容忍前后缀说明文字
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict) and "score" in data:
                return _clamp(float(data["score"]))
        except (ValueError, TypeError):
            pass

    # 第二优先：非严格 JSON 时用正则抓 "score": 0.75
    match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if match:
        return _clamp(float(match.group(1)))

    # 第三优先：智谱 function call 风格输出
    match = re.search(
        r"arg_key[^>]*>\s*score\s*<.*?arg_value[^>]*>\s*([0-9]*\.?[0-9]+)",
        text,
        re.S,
    )
    if match:
        return _clamp(float(match.group(1)))
    # 全部解析失败返回 None，调用方会把它从均值里排除
    return None


def _clamp(value: float) -> float:
    """把分数夹到 [0, 1]，防止模型返回 1.5 之类的越界值。"""
    return max(0.0, min(1.0, value))


def judge_faithfulness(
    question: str,
    answer: str,
    contexts: list[RetrievedChunk],
    llm: LLMClient,
    model: str,
    max_chars: int = 6000,
) -> float | None:
    """用 LLM 裁判判断答案是否被资料支持，返回 0~1 的分数。

    系统正确拒答时直接记 1.0（拒答本身就是忠于资料的表现）；
    离线或裁判输出无法解析时返回 None。
    """
    if is_refusal(answer):
        return 1.0
    if not llm.available:
        return None

    # 每个上下文最多取 1500 字，整体不超过 max_chars
    parts: list[str] = []
    used = 0
    for index, context in enumerate(contexts, start=1):
        text = (context.parent_text or context.child_text)[:1500]
        if used + len(text) > max_chars:
            break
        parts.append(f"[{index}] {text}")
        used += len(text)

    context_text = "\n\n".join(parts)
    prompt = (
        f"<问题>{question}</问题>\n\n"
        f"<回答>{answer}</回答>\n\n"
        f"<资料>\n{context_text}\n</资料>"
    )
    raw = llm.chat(FAITHFULNESS_SYSTEM, prompt, model=model, temperature=0.0)
    score = parse_score(raw)
    if score is None:
        logger.warning("Faithfulness 裁判返回无法解析：%s", (raw or "")[:200])
    return score
