from pathlib import Path

import pytest

from rag_learning.loaders import ParseError, discover_documents, load_document
from rag_learning.loaders.base import AclOverride, infer_acl
from rag_learning.schema import UserContext

DOCS = Path(__file__).resolve().parents[1] / "docs"


def test_all_demo_documents_parse() -> None:
    """10 篇演示文档全部能解析，且覆盖 8 种格式、权限字段合法。"""
    documents = [load_document(path, DOCS) for path in discover_documents(DOCS)]
    assert len(documents) == 10
    formats = {document.doc_format for document in documents}
    assert formats >= {"markdown", "text", "html", "csv", "xlsx", "docx", "pptx", "pdf"}
    for document in documents:
        assert len(document.text) > 100
        assert document.department in {"general", "hr", "finance", "ops", "rd"}
        assert document.sensitivity in {"public", "internal", "confidential"}


def test_tables_are_preserved_as_markdown() -> None:
    """各种格式里的表格都要转成 Markdown 表格，而不是被压成一行文本。"""
    csv_doc = load_document(DOCS / "general/internal/vendor-list.csv", DOCS)
    assert "| 供应商名称 |" in csv_doc.text

    xlsx_doc = load_document(DOCS / "hr/confidential/salary-band.xlsx", DOCS)
    assert "| 职级 |" in xlsx_doc.text
    assert "| 部门 |" in xlsx_doc.text

    docx_doc = load_document(DOCS / "hr/internal/leave-policy.docx", DOCS)
    assert "| 假期类型 |" in docx_doc.text

    html_doc = load_document(DOCS / "ops/public/incident-response.html", DOCS)
    assert "| P0 |" in html_doc.text

    pptx_doc = load_document(DOCS / "rd/internal/api-guidelines.pptx", DOCS)
    assert "| 40001 |" in pptx_doc.text


def test_pdf_title_comes_from_metadata() -> None:
    """PDF 标题优先取文档元数据，而不是文件名。"""
    document = load_document(DOCS / "finance/internal/expense-reimbursement.pdf", DOCS)
    assert document.title == "费用报销管理办法"
    assert "发票抬头" in document.text


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
