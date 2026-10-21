from pathlib import Path

import pytest

from rag_learning.loaders import ParseError, discover_documents, load_document
from rag_learning.loaders.base import AclOverride, infer_acl
from rag_learning.schema import UserContext

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_all_corpus_documents_parse() -> None:
    """语料库全部文档都能解析，且权限字段合法。"""
    documents = [load_document(path, DOCS) for path in discover_documents(DOCS)]
    assert len(documents) >= 41
    for document in documents:
        assert len(document.text) > 50
        assert document.department in {"general", "hr", "finance", "ops", "rd"}
        assert document.sensitivity in {"public", "internal", "confidential"}


def test_corpus_covers_all_formats() -> None:
    """语料需要覆盖 md / docx / pdf / html / pptx 五种格式。"""
    documents = [load_document(path, DOCS) for path in discover_documents(DOCS)]
    formats = {document.doc_format for document in documents}
    assert formats >= {"markdown", "docx", "pdf", "html", "pptx"}


def test_tables_are_preserved_as_markdown() -> None:
    """含地址表格的分行网络文档，解析后必须保留表格字段。

    只挑真正的分行网络类文档（branch-network / branches-in），它们的正文
    一定是 IFSC 代码表格；ATM 列表类文档不含表格，不在此断言范围内。
    """
    checked = 0
    for path in discover_documents(DOCS):
        stem = path.stem.lower()
        if not ("branch" in stem or "network" in stem):
            continue
        if "atm" in stem:
            continue
        document = load_document(path, DOCS)
        assert "IFSC" in document.text
        checked += 1
    assert checked >= 3


def test_pdf_chinese_text_is_extractable() -> None:
    """PDF 里的中文必须能被 pypdfium2 提取，不能是乱码或空文本。"""
    document = load_document(DOCS / "finance/internal/personal-loan.pdf", DOCS)
    assert "贷款" in document.text
    assert len(document.text) > 200
    # 不应出现替换字符（乱码的典型表现）
    assert "\ufffd" not in document.text


def test_docx_headings_become_markdown() -> None:
    """DOCX 的 Heading 样式要转成 Markdown 标题，供 chunker 识别层级。"""
    document = load_document(DOCS / "finance/internal/zx-bank-business-loans.docx", DOCS)
    assert "#" in document.text
    assert "贷款" in document.text


def test_acl_precedence() -> None:
    """权限优先级：override > frontmatter > 目录 > 默认值。"""
    department, sensitivity, source = infer_acl(
        Path("hr/public/note.md"),
        {"department": "finance", "sensitivity": "confidential"},
    )
    assert (department, sensitivity, source) == ("finance", "confidential", "frontmatter")

    department, sensitivity, source = infer_acl(Path("hr/internal/note.txt"), {})
    assert (department, sensitivity, source) == ("hr", "internal", "directory")

    department, sensitivity, source = infer_acl(Path("unknown/note.md"), {})
    assert (department, sensitivity, source) == ("general", "public", "default")

    department, sensitivity, source = infer_acl(
        Path("hr/internal/note.md"),
        {"department": "hr", "sensitivity": "internal"},
        AclOverride(department="ops", sensitivity="public"),
    )
    assert (department, sensitivity, source) == ("ops", "public", "override")


def test_clearance_inheritance() -> None:
    """密级向下兼容：internal 能看 public + internal，不能看 confidential。"""
    user = UserContext("hr", "internal")
    assert user.visible_sensitivities == ["public", "internal"]
    assert user.can_access("general", "public")
    assert user.can_access("hr", "internal")
    assert not user.can_access("hr", "confidential")
    assert not user.can_access("ops", "public")


def test_scanned_pdf_raises(tmp_path: Path) -> None:
    """没有任何文字的空 PDF 应判定为扫描件并报错，而不是静默入库空文本。"""
    from reportlab.pdfgen import canvas

    path = tmp_path / "blank.pdf"
    canvas.Canvas(str(path)).save()
    with pytest.raises(ParseError):
        load_document(path, tmp_path)


def test_unknown_format_raises(tmp_path: Path) -> None:
    """不支持的扩展名要明确报 ParseError。"""
    path = tmp_path / "note.xyz"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ParseError):
        load_document(path, tmp_path)
