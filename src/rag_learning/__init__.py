"""包入口：console script 和 python -m 都走这里。"""

from __future__ import annotations


def main() -> None:
    """延迟导入 cli，避免 import rag_learning 时就加载 typer 等重依赖。"""
    from .cli import main as _main

    _main()


__all__ = ["main"]
