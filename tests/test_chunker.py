from rag_learning.chunker import MAX_TABLE_BYTES, chunk_document, compute_pk


def _make_table(rows: int, cell: str = "内容") -> str:
    """造一张带表头分隔行的 Markdown 表格，行数由参数控制。"""
    header = "| 职级 | 下限 | 上限 | 说明 |\n| --- | --- | --- | --- |"
    body = "\n".join(f"| P{i} | {1000 + i} | {2000 + i} | {cell * 8} |" for i in range(rows))
    return f"{header}\n{body}"


def test_large_table_stays_atomic() -> None:
    """1500+ 字符的表格必须整体成为一个块，不能被长度切分拆开。"""
    table = _make_table(40)
    assert len(table) > 1500
    chunks = chunk_document(f"# 薪酬带宽\n{table}", "doc-table")
    assert len(chunks) == 1
    # 子块带标题前缀，父块保留完整表格
    assert chunks[0].child_text.startswith("【薪酬带宽】")
    assert "P39" in chunks[0].child_text
    assert chunks[0].parent_text == table


def test_table_over_milvus_limit_is_split_with_warning() -> None:
    """超过 Milvus 字节上限的表格允许降级切分，但每块都要重复表头。"""
    table = _make_table(2000, cell="很长的内容")
    assert len(table.encode("utf-8")) > MAX_TABLE_BYTES
    chunks = chunk_document(f"# 大表\n{table}", "doc-big-table")
    assert len(chunks) > 1
    assert all(chunk.parent_text.startswith("| 职级 |") for chunk in chunks)


def test_parent_child_structure() -> None:
    """长文本要切出多个父块，且每个子块都能追溯到自己父块的标题。"""
    text = "# 标题\n" + "\n\n".join(f"第{i}段内容，讲的是数据库连接池的排查方法。" for i in range(80))
    chunks = chunk_document(text, "doc-long", parent_size=600, child_size=120, chunk_overlap=20)
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.parent_text
        assert chunk.child_text.startswith("【标题】")
        assert chunk.heading == "标题"
    assert len({chunk.parent_id for chunk in chunks}) > 1


def test_deterministic_primary_key() -> None:
    """同一文档重复切分主键必须一致，不同文档必须不同（保证 upsert 幂等）。"""
    text = "第一段内容。\n\n第二段内容。"
    first = chunk_document(text, "doc-a")
    second = chunk_document(text, "doc-a")
    other = chunk_document(text, "doc-b")
    assert [chunk.pk for chunk in first] == [chunk.pk for chunk in second]
    assert [chunk.pk for chunk in first] != [chunk.pk for chunk in other]
    assert first[0].pk == compute_pk("doc-a", 0)


def test_chunk_index_is_continuous() -> None:
    """chunk_index 必须从 0 连续递增，delete_tail 依赖这个前提。"""
    text = "# 标题\n" + "\n\n".join(f"第{i}段。" for i in range(50))
    chunks = chunk_document(text, "doc-index", parent_size=200, child_size=60)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
