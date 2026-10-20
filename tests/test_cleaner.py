from rag_learning.cleaner import clean_query, clean_text


def test_clean_text_idempotent() -> None:
    """清洗必须幂等：对清洗结果再清洗一次，结果不变。"""
    raw = "  第一行\r\n\r\n\r\n第二行\u200b 有零宽  \t 字符  "
    once = clean_text(raw)
    assert clean_text(once) == once
    assert "\r" not in once
    assert "\u200b" not in once
    assert "\n\n\n" not in once


def test_clean_text_removes_zero_width_between_chinese() -> None:
    """中文之间的零宽字符要去掉，否则会改变分词结果。"""
    assert clean_text("连\u200b接\u200c池") == "连接池"


def test_clean_text_collapses_spaces() -> None:
    """行内连续空格和制表符压成一个空格。"""
    assert clean_text("a   b\t\tc") == "a b c"


def test_clean_query_single_line() -> None:
    """查询端清洗后必须是单行，避免改写模型收到碎行。"""
    assert clean_query("连接池\n\n耗尽了   怎么办") == "连接池 耗尽了 怎么办"
