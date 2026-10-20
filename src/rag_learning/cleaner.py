"""文本清洗：文档端和查询端共用同一套规则。"""

from __future__ import annotations

import re

# 零宽字符：从网页/Office 复制文本时经常混进来，会污染稀疏检索的 token
ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\u00ad]")
# 不可见控制字符：保留 \t \n \r，其余删掉，避免破坏 Milvus 字符串字段
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# 行内连续空白压缩成一个空格
INLINE_SPACE_RE = re.compile(r"[ \t]+")
# 行首行尾多余空格，连同换行一起收拢
LINE_EDGE_SPACE_RE = re.compile(r" *\n *")
# 连续 3 个以上换行压成空行，保证幂等（clean(clean(x)) == clean(x)）
MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """统一换行、去零宽/控制字符、压缩空白。保证幂等。"""
    if not text:
        return ""
    # 先统一换行符，后续所有正则只需要处理 \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = ZERO_WIDTH_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    # 不换行空格和全角空格统一成半角空格
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = INLINE_SPACE_RE.sub(" ", text)
    text = LINE_EDGE_SPACE_RE.sub("\n", text)
    text = MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def clean_query(text: str) -> str:
    """查询端清洗：额外把换行压成空格，避免改写模型收到碎行。"""
    return " ".join(clean_text(text).split())
