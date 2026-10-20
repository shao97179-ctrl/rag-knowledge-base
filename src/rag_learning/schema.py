"""统一的领域数据模型，避免各模块各发明一套字典。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 密级用整数表示，便于做「不高于我的密级」的集合运算
SENSITIVITY_LEVELS: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
}
# 部门是穷举的枚举；general 表示全公司可见
DEPARTMENTS: tuple[str, ...] = ("general", "hr", "finance", "ops", "rd")


def normalize_department(value: str) -> str:
    """把部门名归一化为小写并校验，非法值抛出 ValueError。"""
    value = (value or "").strip().lower()
    if value not in DEPARTMENTS:
        raise ValueError(f"未知部门：{value!r}，可选值：{', '.join(DEPARTMENTS)}")
    return value


def normalize_sensitivity(value: str) -> str:
    """把密级名归一化为小写并校验，非法值抛出 ValueError。"""
    value = (value or "").strip().lower()
    if value not in SENSITIVITY_LEVELS:
        raise ValueError(f"未知密级：{value!r}，可选值：{', '.join(SENSITIVITY_LEVELS)}")
    return value


@dataclass(frozen=True)
class UserContext:
    """一次查询的身份：部门 + 密级。演示用，不是生产鉴权。"""

    department: str = "general"
    clearance: str = "public"

    def __post_init__(self) -> None:
        """frozen dataclass 只能用 object.__setattr__ 写回归一化后的值。"""
        object.__setattr__(self, "department", normalize_department(self.department))
        object.__setattr__(self, "clearance", normalize_sensitivity(self.clearance))

    @property
    def level(self) -> int:
        """当前密级对应的数值等级，用于比较可见范围。"""
        return SENSITIVITY_LEVELS[self.clearance]

    @property
    def visible_sensitivities(self) -> list[str]:
        """当前用户可见的密级列表（等级不高于自身 clearance）。"""
        return [name for name, level in SENSITIVITY_LEVELS.items() if level <= self.level]

    def can_access(self, department: str, sensitivity: str) -> bool:
        """判断某个 chunk 的「部门 + 密级」组合是否对当前用户可见。

        规则：部门等于自己的部门或 general，且密级不超过自己的密级。
        """
        department = department.strip().lower()
        sensitivity = sensitivity.strip().lower()
        return (
            department in {self.department, "general"}
            and sensitivity in self.visible_sensitivities
        )


@dataclass
class ParsedDocument:
    """loader 的输出：清洗前/后的正文 + 权限元数据。"""

    doc_id: str
    source_path: Path
    rel_path: str
    title: str
    doc_format: str
    text: str
    department: str
    sensitivity: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """检索单元：子块进索引，父块用于生成上下文。"""

    pk: str
    doc_id: str
    chunk_index: int
    parent_id: str
    parent_index: int
    child_text: str
    parent_text: str
    heading: str


@dataclass
class RetrievedChunk:
    """检索结果：在 Chunk 基础上附加分数和两路排名，供 UI / 评估观测。"""

    pk: str
    doc_id: str
    chunk_index: int
    parent_id: str
    child_text: str
    parent_text: str
    heading: str
    title: str
    source_path: str
    department: str
    sensitivity: str
    doc_format: str
    score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None


@dataclass
class Citation:
    """返回给用户的引用卡片，index 与答案里的 [n] 对应。"""

    index: int
    title: str
    source_path: str
    heading: str
    department: str
    sensitivity: str
    snippet: str
    score: float


@dataclass
class AnswerResult:
    """一次问答的完整结果，API / CLI / 评估共用。"""

    answer: str
    citations: list[Citation]
    contexts: list[RetrievedChunk]
    rewritten_query: str
    refused: bool
    offline: bool
    debug: dict[str, Any] = field(default_factory=dict)
