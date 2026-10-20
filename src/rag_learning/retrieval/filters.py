"""ACL 过滤表达式：权限过滤必须发生在检索层。"""

from __future__ import annotations

import json

from ..schema import UserContext


def build_acl_expr(user: UserContext) -> str:
    """生成 Milvus 的 filter 表达式，例如：

    department in ["ops", "general"] and sensitivity in ["public", "internal"]

    general 部门对所有用户可见；密级取用户可见的所有级别（含更低级别）。
    用 json.dumps 生成列表字面量，顺便处理引号和转义。
    """
    departments = [user.department]
    if user.department != "general":
        departments.append("general")
    return (
        f"department in {json.dumps(departments, ensure_ascii=False)} "
        f"and sensitivity in {json.dumps(user.visible_sensitivities, ensure_ascii=False)}"
    )
