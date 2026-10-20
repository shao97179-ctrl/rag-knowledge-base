"""日志配置：只允许入口（CLI / API）调用 configure_logging。"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
# 这些库的 INFO 日志很吵，统一压到 WARNING
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "transformers",
    "pymilvus",
    "milvus_lite",
    "openai",
)


def configure_logging(level: str = "INFO") -> None:
    """配置根 logger，重复调用不会叠加 handler。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # milvus-lite 会对未实现的 AllocTimestamp 打 ERROR，但检索仍正常，压掉噪音
    logging.getLogger("grpc").setLevel(logging.CRITICAL)
    _CONFIGURED = True
