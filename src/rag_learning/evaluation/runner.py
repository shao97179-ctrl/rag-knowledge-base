"""评估入口：跑黄金集、打印指标、写报告、按阈值返回退出码。"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.table import Table

from ..config import PROJECT_ROOT, get_settings
from ..loaders.base import doc_id_for
from ..pipeline import get_pipeline
from ..schema import UserContext
from .metrics import judge_faithfulness, mrr_at_k, recall_at_k

logger = logging.getLogger(__name__)
console = Console(highlight=False)


@dataclass
class CaseResult:
    """单条黄金集用例的评估结果。"""

    id: str
    case_type: str
    question: str
    answer: str
    recall: float | None = None
    mrr: float | None = None
    faithfulness: float | None = None
    acl_violations: list[dict] = field(default_factory=list)
    forbidden_hits: list[str] = field(default_factory=list)
    refused: bool | None = None
    refusal_ok: bool | None = None
    top_paths: list[str] = field(default_factory=list)
    error: str | None = None


def load_cases(path: Path) -> list[dict]:
    """逐行读取 JSONL 黄金集，跳过空行和 # 注释行。"""
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"黄金测试集不存在：{path}")
    cases: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"黄金集第 {line_number} 行不是合法 JSON：{exc}") from exc
    return cases


def _violation_records(result, user: UserContext) -> list[dict]:
    """收集本次用例里的 ACL 违规：既查最终上下文，也查两路原始召回。

    查原始召回是为了发现「filter 写错但恰好没进最终 top-k」这类隐患。
    """
    records: list[dict] = []
    for chunk in result.contexts:
        if not user.can_access(chunk.department, chunk.sensitivity):
            records.append(
                {
                    "source_path": chunk.source_path,
                    "department": chunk.department,
                    "sensitivity": chunk.sensitivity,
                }
            )
    retrieval = result.debug.get("retrieval", {})
    for key in ("dense_hits", "sparse_hits"):
        for hit in retrieval.get(key, []):
            if not user.can_access(hit["department"], hit["sensitivity"]):
                records.append(
                    {
                        "source_path": hit["source_path"],
                        "department": hit["department"],
                        "sensitivity": hit["sensitivity"],
                    }
                )
    # 同一文档可能被多路多次命中，按 (路径, 部门, 密级) 去重
    deduped: dict[tuple, dict] = {}
    for record in records:
        key = (record["source_path"], record["department"], record["sensitivity"])
        deduped.setdefault(key, record)
    return list(deduped.values())


def run_evaluation(
    golden_path: Path,
    no_faithfulness: bool = False,
    limit: int | None = None,
) -> int:
    """跑完整黄金集，打印明细和汇总，写 JSON 报告，返回进程退出码。"""
    settings = get_settings()
    pipeline = get_pipeline()
    cases = load_cases(golden_path)
    if limit is not None:
        cases = cases[:limit]

    results: list[CaseResult] = []
    # Recall / MRR 的 K 跟随检索配置：top_k_final 是几就评 @几，标签和口径一致
    metric_k = settings.top_k_final
    for case in cases:
        case_id = case.get("id", "?")
        case_type = case.get("type", "retrieval")
        question = case["question"]
        user = UserContext(case.get("department", "general"), case.get("clearance", "public"))
        try:
            answer = asyncio.run(pipeline.ask(question, user))
            # 按父块顺序去重后取文档路径，Recall / MRR 以文档为单位计算
            retrieved_paths = list(dict.fromkeys(context.source_path for context in answer.contexts))
            retrieved_ids = [doc_id_for(path) for path in retrieved_paths]
            expected_ids = [doc_id_for(path) for path in case.get("expected_paths", [])]

            recall = None
            mrr = None
            # 只有 retrieval 类型且声明了预期文档才计算检索指标
            if case_type == "retrieval" and expected_ids:
                recall = recall_at_k(retrieved_ids, expected_ids, k=metric_k)
                mrr = mrr_at_k(retrieved_ids, expected_ids, k=metric_k)

            violations = _violation_records(answer, user)
            forbidden = sorted(set(retrieved_paths) & set(case.get("forbidden_paths", [])))

            faithfulness = None
            # ACL 用例不评 Faithfulness：它的目的是验证越权，不是答案质量
            if (
                not no_faithfulness
                and pipeline.llm.available
                and case_type in {"retrieval", "refusal"}
            ):
                faithfulness = judge_faithfulness(
                    question,
                    answer.answer,
                    answer.contexts,
                    pipeline.llm,
                    settings.llm_model,
                    settings.max_context_chars,
                )
                if faithfulness is None:
                    logger.warning("用例 %s 的 Faithfulness 无法解析，已排除", case_id)

            refusal_ok = None
            # 拒答能力依赖真实 LLM，离线模式无法判断
            if case_type == "refusal" and pipeline.llm.available:
                refusal_ok = bool(answer.refused)

            results.append(
                CaseResult(
                    id=case_id,
                    case_type=case_type,
                    question=question,
                    answer=answer.answer,
                    recall=recall,
                    mrr=mrr,
                    faithfulness=faithfulness,
                    acl_violations=violations,
                    forbidden_hits=forbidden,
                    refused=bool(answer.refused),
                    refusal_ok=refusal_ok,
                    top_paths=retrieved_paths[:3],
                )
            )
        except Exception as exc:  # noqa: BLE001
            # 单条用例失败不中断整轮评估，记下来最后统一退出码 1
            logger.exception("用例 %s 执行失败", case_id)
            results.append(
                CaseResult(
                    id=case_id,
                    case_type=case_type,
                    question=question,
                    answer="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    _print_table(results, metric_k)
    return _summarize(results, settings, no_faithfulness, metric_k)


def _print_table(results: list[CaseResult], metric_k: int) -> None:
    """用 rich 打印逐条明细表，有 ACL 违规的行标红。"""
    table = Table(title="评估明细")
    table.add_column("用例")
    table.add_column("类型")
    table.add_column(f"Recall@{metric_k}", justify="right")
    table.add_column(f"MRR@{metric_k}", justify="right")
    table.add_column("Faith", justify="right")
    table.add_column("ACL", justify="right")
    table.add_column("Top1", overflow="fold")
    for result in results:
        acl = len(result.acl_violations) + len(result.forbidden_hits)
        table.add_row(
            result.id,
            result.case_type,
            "-" if result.recall is None else f"{result.recall:.2f}",
            "-" if result.mrr is None else f"{result.mrr:.2f}",
            "-" if result.faithfulness is None else f"{result.faithfulness:.2f}",
            str(acl),
            result.error or (result.top_paths[0] if result.top_paths else "-"),
            style="red" if acl else "white",
        )
    console.print(table)


def _summarize(results: list[CaseResult], settings, no_faithfulness: bool, metric_k: int) -> int:
    """汇总指标、写 JSON 报告，并按阈值返回退出码（0/1/2）。"""
    # None 表示该用例不适用（例如离线模式没有 Faithfulness），从均值里排除
    recalls = [result.recall for result in results if result.recall is not None]
    mrrs = [result.mrr for result in results if result.mrr is not None]
    faiths = [result.faithfulness for result in results if result.faithfulness is not None]
    acl_total = sum(len(result.acl_violations) + len(result.forbidden_hits) for result in results)
    refusal_cases = [result for result in results if result.refusal_ok is not None]
    refusal_ok = sum(1 for result in refusal_cases if result.refusal_ok)
    errors = [result for result in results if result.error]

    recall_avg = mean(recalls) if recalls else None
    mrr_avg = mean(mrrs) if mrrs else None
    faith_avg = mean(faiths) if faiths else None

    console.print()
    console.print(f"[bold]Recall@{metric_k}[/bold] = {recall_avg:.3f}" if recall_avg is not None else f"Recall@{metric_k} = N/A")
    console.print(f"[bold]MRR@{metric_k}[/bold]    = {mrr_avg:.3f}" if mrr_avg is not None else f"MRR@{metric_k} = N/A")
    if faith_avg is not None:
        console.print(f"[bold]Faithfulness[/bold] = {faith_avg:.3f}（{len(faiths)} 条）")
    elif no_faithfulness:
        console.print("Faithfulness = N/A（已用 --no-faithfulness 跳过）")
    elif not settings.llm_available:
        console.print("Faithfulness = N/A（未配置 API Key，未调用裁判）")
    else:
        console.print("Faithfulness = N/A（裁判结果全部无法解析）")
    console.print(f"[bold]ACL 违规[/bold] = {acl_total}")
    if refusal_cases:
        console.print(f"[bold]拒答正确[/bold] = {refusal_ok}/{len(refusal_cases)}")
    else:
        console.print("拒答正确 = N/A（需要真实 LLM）")
    if errors:
        console.print(f"[red]执行失败用例：{[result.id for result in errors]}[/red]")

    # 报告落到 eval/reports/，文件名带时间戳，方便对比不同版本
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "metrics": {
            f"recall_at_{metric_k}": recall_avg,
            f"mrr_at_{metric_k}": mrr_avg,
            "faithfulness": faith_avg,
            "acl_violations": acl_total,
            "refusal_ok": refusal_ok,
            "refusal_total": len(refusal_cases),
            "errors": len(errors),
        },
        "cases": [asdict(result) for result in results],
    }
    reports_dir = PROJECT_ROOT / "eval" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"报告已写入：{report_path}")

    # 退出码约定：2 = ACL 违规（CI 硬阻断），1 = 质量不达标，0 = 通过
    if acl_total > 0:
        console.print("[red]存在 ACL 违规，退出码 2[/red]")
        return 2
    if recall_avg is not None and recall_avg < 0.85:
        console.print(f"[red]Recall@{metric_k} 低于 0.85[/red]")
        return 1
    if faith_avg is not None and faith_avg < 0.9:
        console.print("[red]Faithfulness 低于 0.9[/red]")
        return 1
    if refusal_cases and refusal_ok < len(refusal_cases):
        console.print("[red]存在拒答失败用例[/red]")
        return 1
    if errors:
        return 1
    return 0
