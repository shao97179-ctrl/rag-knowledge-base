"""OpenAI 兼容 LLM 客户端：智谱 / DeepSeek / 离线三种模式。"""

from __future__ import annotations

import logging
import time
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """统一封装 chat 调用；失败返回 None，由上层决定如何降级。"""

    def __init__(self, settings: Settings) -> None:
        """记录 provider 和可用性；不可用时只打一条 WARNING，不抛异常。"""
        self.settings = settings
        self.provider = settings.llm_provider
        self.api_key = settings.api_key
        self.available = settings.llm_available
        # 按模型名缓存 client，主模型和轻量改写模型各一个
        self._clients: dict[str, ChatOpenAI] = {}
        if not self.available:
            logger.warning("LLM 不可用（provider=%s），进入离线抽取式模式", self.provider)

    def _client(self, model: str | None = None) -> ChatOpenAI | None:
        """取（或懒创建）某个模型的 client；离线模式返回 None。"""
        if not self.available:
            return None
        model = model or self.settings.llm_model
        if model not in self._clients:
            self._clients[model] = ChatOpenAI(
                model=model,
                api_key=self.api_key,
                base_url=self.settings.llm_base_url,
                temperature=self.settings.llm_temperature,
                timeout=self.settings.llm_timeout,
                max_retries=1,
            )
        return self._clients[model]

    def chat(
        self,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """返回文本；失败返回 None，由调用方走降级路径。"""
        client = self._client(model)
        if client is None:
            return None
        started = time.perf_counter()
        try:
            response = client.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            content = response.content
            # 部分模型返回 content 列表（多模态分片），这里拼成纯文本
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            text = str(content).strip()
            logger.info(
                "LLM 调用成功 provider=%s model=%s elapsed=%.2fs chars=%s",
                self.provider,
                model or self.settings.llm_model,
                time.perf_counter() - started,
                len(text),
            )
            return text or None
        except Exception as exc:  # noqa: BLE001
            # 任何异常都不上抛：上层统一走抽取式降级，保证问答不中断
            logger.warning(
                "LLM 调用失败 provider=%s model=%s elapsed=%.2fs error=%s",
                self.provider,
                model or self.settings.llm_model,
                time.perf_counter() - started,
                type(exc).__name__,
            )
            return None


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """进程内单例，避免每次请求都重新构造 client。"""
    return LLMClient(get_settings())
