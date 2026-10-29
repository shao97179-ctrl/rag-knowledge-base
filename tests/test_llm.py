"""LLMClient 离线测试：不发起真实请求，只验证 client 的构造和缓存逻辑。"""

import asyncio
from types import SimpleNamespace

from rag_learning.llm import LLMClient


def _settings(**overrides) -> SimpleNamespace:
    """LLMClient 用到的最小配置桩。"""
    base = {
        "llm_provider": "zhipu",
        "llm_base_url": "http://localhost:9/v1",
        "llm_model": "main-model",
        "llm_temperature": 0.1,
        "llm_timeout": 1,
        "api_key": "test-key",
        "llm_available": True,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_temperature_is_applied_and_clients_cached() -> None:
    """chat 传入的 temperature 必须传到 client；同参数复用同一个实例。"""
    llm = LLMClient(_settings())
    default = llm._client()
    zero = llm._client(temperature=0.0)

    assert default is not zero
    assert default.temperature == 0.1
    assert zero.temperature == 0.0
    # 缓存键是 (model, temperature)，同参数拿回同一个对象
    assert llm._client() is default
    assert llm._client(temperature=0.0) is zero


def test_offline_client_is_none() -> None:
    """离线模式下不构造任何 client。"""
    llm = LLMClient(_settings(llm_provider="offline", api_key=None, llm_available=False))
    assert llm._client() is None
    assert llm._client(temperature=0.0) is None
    # 异步路径同样直接返回 None，不发起调用
    assert asyncio.run(llm.achat("sys", "user")) is None
