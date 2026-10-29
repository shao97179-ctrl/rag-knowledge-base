"""多轮查询改写：只在有历史时调用轻量模型，单轮不改写。"""

from __future__ import annotations

import logging

from ..cleaner import clean_query
from ..llm import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个查询改写助手。根据对话历史，把用户的最新问题改写成不依赖上下文、可独立检索的完整问题。
要求：
1. 保留原问题的关键实体和限定条件；
2. 不要回答问题，不要解释；
3. 只输出改写后的问题本身，一行文本。"""


class QueryRewriter:
    """把带指代的多轮问题改写成可独立检索的查询。"""

    def __init__(self, llm: LLMClient, light_model: str) -> None:
        """改写用轻量模型，成本和延迟都更低。"""
        self.llm = llm
        self.light_model = light_model

    async def rewrite(self, query: str, history: list[dict] | None) -> str:
        """有历史且 LLM 可用时改写；任何一步失败都退回原问题。"""
        original = clean_query(query)
        # 单轮对话没有指代问题，直接返回，省一次 API 调用
        if not history:
            return original
        if not self.llm.available:
            return original

        # 只取最近 6 条，每条截断 300 字，控制 prompt 长度
        history_text = "\n".join(
            f"{'用户' if item.get('role') == 'user' else '助手'}：{item.get('content', '')[:300]}"
            for item in history[-6:]
        )
        user_prompt = f"对话历史：\n{history_text}\n\n最新问题：{original}\n\n改写后的问题："
        result = await self.llm.achat(
            SYSTEM_PROMPT,
            user_prompt,
            model=self.light_model,
            temperature=0.0,
        )
        if not result:
            logger.warning("查询改写失败，使用原问题")
            return original
        # 模型偶尔会加引号或多输出几行，只取第一行并去掉包裹的引号
        rewritten = clean_query(result.splitlines()[0].strip().strip('"').strip("'"))
        if not rewritten:
            return original
        logger.info("查询改写：%s -> %s", original, rewritten)
        return rewritten
