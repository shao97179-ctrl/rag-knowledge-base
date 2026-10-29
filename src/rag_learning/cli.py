"""命令行入口：ingest / ask / eval / probe / serve / stats。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .config import get_settings
from .embedder import get_embedder
from .ingest import Ingestor, IngestSummary
from .loaders import AclOverride
from .logging_config import configure_logging
from .pipeline import get_pipeline
from .registry import Registry
from .schema import AnswerResult, UserContext, normalize_department, normalize_sensitivity
from .vector_store import MilvusStore

app = typer.Typer(add_completion=False, help="企业知识库 RAG：入库、问答、评估、自检")
console = Console(highlight=False)

# 入库状态到颜色的映射，表格里一眼能看出问题
_STATUS_STYLE = {
    "indexed": "green",
    "unchanged": "dim",
    "deleted": "yellow",
    "empty": "yellow",
    "failed": "red",
}


def _user(department: str, clearance: str) -> UserContext:
    """校验命令行传入的部门/密级，非法值转成 typer 的参数错误。"""
    try:
        normalize_department(department)
        normalize_sensitivity(clearance)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return UserContext(department, clearance)


def _print_summary(summary: IngestSummary) -> None:
    """打印逐文件入库结果表，有失败时额外提示看日志。"""
    table = Table(title=f"入库结果（indexed={summary.indexed} unchanged={summary.unchanged}）")
    table.add_column("文件", overflow="fold")
    table.add_column("状态")
    table.add_column("格式")
    table.add_column("部门/密级")
    table.add_column("块数", justify="right")
    for result in summary.results:
        table.add_row(
            result.rel_path,
            result.status,
            result.doc_format or "-",
            f"{result.department}/{result.sensitivity}" if result.department else "-",
            str(result.chunks),
            style=_STATUS_STYLE.get(result.status, "white"),
        )
    console.print(table)
    if summary.failed:
        console.print("[red]有文件入库失败，请查看上面的 ERROR 日志。[/red]")


def _print_answer(result: AnswerResult, debug: bool) -> None:
    """打印答案、离线提示、引用表和可选的调试 JSON。"""
    console.print()
    console.print(Text(result.answer))
    if result.offline:
        console.print("[yellow]（当前为离线/降级模式，未调用主 LLM）[/yellow]")
    if result.citations:
        table = Table(title="引用来源")
        table.add_column("#", justify="right")
        table.add_column("文件", overflow="fold")
        table.add_column("标题", overflow="fold")
        table.add_column("权限")
        table.add_column("章节", overflow="fold")
        table.add_column("RRF", justify="right")
        for citation in result.citations:
            table.add_row(
                str(citation.index),
                citation.source_path,
                citation.title,
                f"{citation.department}/{citation.sensitivity}",
                citation.heading or "-",
                f"{citation.score:.5f}",
            )
        console.print(table)
    if debug:
        console.print(JSON(json.dumps(result.debug, ensure_ascii=False, default=str)))


@app.command()
def ingest(
    path: Optional[str] = typer.Argument(None, help="只入库单个文件；留空则扫描 docs"),
    force: bool = typer.Option(False, "--force", help="重建集合和索引（幂等）"),
    department: Optional[str] = typer.Option(None, "--department", help="覆盖部门"),
    sensitivity: Optional[str] = typer.Option(None, "--sensitivity", help="覆盖密级"),
) -> None:
    """解析文档并写入 Milvus；未变化的文档自动跳过。"""
    # Ingestor 里还有一层兜底，这里先拦住是为了给用户干净的报错而不是 traceback
    if force and path:
        raise typer.BadParameter(
            "--force 会清空整个索引后重建，不能与文件参数同时使用；"
            "需要重建单篇文档请直接全量 ingest --force，或去掉 --force 单独入库该文件"
        )
    settings = get_settings()
    override = None
    if department or sensitivity:
        # 先校验再构造，避免非法值一路传到写库阶段
        try:
            if department:
                normalize_department(department)
            if sensitivity:
                normalize_sensitivity(sensitivity)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        override = AclOverride(department=department, sensitivity=sensitivity)

    embedder = get_embedder(settings)
    store = MilvusStore(settings.milvus_uri, settings.milvus_collection, embedder.DIMENSION)
    registry = Registry(settings.registry_path)
    try:
        ingestor = Ingestor(settings, embedder, store, registry)
        summary = ingestor.ingest(
            paths=[Path(path)] if path else None,
            force=force,
            override=override,
        )
        _print_summary(summary)
        if summary.failed:
            # 有失败文件时以非零退出码结束，方便脚本判断
            raise typer.Exit(1)
    finally:
        store.close()


@app.command()
def ask(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="单次问题；留空进入交互模式"),
    dept: str = typer.Option("general", "--dept", help="部门：general/hr/finance/ops/rd"),
    clearance: str = typer.Option("public", "--clearance", help="密级：public/internal/confidential"),
    debug: bool = typer.Option(False, "--debug", help="输出检索调试信息"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="最终父块数量"),
) -> None:
    """提问并返回带引用的答案；留空进入多轮交互模式。"""
    user = _user(dept, clearance)
    pipeline = get_pipeline()

    if query:
        result = asyncio.run(pipeline.ask(query, user, top_k=top_k))
        _print_answer(result, debug)
        return

    console.print("交互模式：输入问题，输入 exit 退出。")
    history: list[dict] = []
    while True:
        try:
            question = Prompt.ask("[bold cyan]你[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"exit", "quit", "q"}:
            break
        if not question.strip():
            continue
        # 交互模式下每轮都把历史带上，触发多轮查询改写
        result = asyncio.run(pipeline.ask(question, user, history, top_k))
        _print_answer(result, debug)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result.answer})


@app.command(name="eval")
def evaluate(
    golden: Path = typer.Option(Path("eval/golden_set.jsonl"), "--golden", help="黄金测试集"),
    no_faithfulness: bool = typer.Option(False, "--no-faithfulness", help="跳过 LLM 裁判"),
    limit: Optional[int] = typer.Option(None, "--limit", help="只跑前 N 条"),
) -> None:
    """跑黄金测试集；ACL 违规时退出码为 2。"""
    # 延迟导入：eval 依赖 pipeline，避免其他命令也拉起整条链路
    from .evaluation.runner import run_evaluation

    raise typer.Exit(run_evaluation(golden, no_faithfulness=no_faithfulness, limit=limit))


@app.command()
def probe(
    live: bool = typer.Option(False, "--live", help="真实调用 LLM（会产生费用）"),
) -> None:
    """验证 BGE-M3、milvus-lite 和 LLM 是否正常。"""
    from .probe import run_probe

    raise typer.Exit(run_probe(live))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """启动 FastAPI + Web UI。"""
    import uvicorn

    uvicorn.run("rag_learning.api:app", host=host, port=port, reload=reload)


@app.command()
def stats() -> None:
    """查看模型、向量库和文档注册表状态。"""
    pipeline = get_pipeline()
    console.print(JSON(json.dumps(pipeline.health(), ensure_ascii=False, default=str)))

    records = pipeline.registry.list_all()
    table = Table(title="已入库文档")
    table.add_column("文件", overflow="fold")
    table.add_column("部门/密级")
    table.add_column("格式")
    table.add_column("块数", justify="right")
    table.add_column("状态")
    for record in records:
        table.add_row(
            record.rel_path,
            f"{record.department}/{record.sensitivity}",
            record.doc_format,
            str(record.chunk_count),
            record.status,
            style=_STATUS_STYLE.get(record.status, "white"),
        )
    console.print(table)


def main() -> None:
    """console script 入口：先配置日志，再交给 typer。"""
    configure_logging(get_settings().log_level)
    app()


if __name__ == "__main__":
    main()
