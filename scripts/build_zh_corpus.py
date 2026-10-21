"""把中文 Markdown 语料生成 5 种格式（md / docx / pdf / html / pptx）并回读校验。

翻译不在这里做：译文由会话中的模型直接产出，写入 zh_src/ 后由本脚本生成其余格式。
原数据的 5 种格式是同一内容的副本，所以只需一份中文 md 就能生成全部格式。

用法：
    .venv/bin/python scripts/build_zh_corpus.py            # 处理 zh_src 下全部文件
    .venv/bin/python scripts/build_zh_corpus.py --only a b # 只处理指定 stem
    .venv/bin/python scripts/build_zh_corpus.py --verify   # 只校验已生成结果

注意：不读取 .env，不调用任何 LLM，不消耗 API 额度。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 项目根：本文件在 scripts/ 下，向上两级即仓库根（与 config.PROJECT_ROOT 同一套路）
ROOT = Path(__file__).resolve().parents[1]

# 中文原文目录：文件为 <部门>__<密级>__<stem>.md
SRC = ROOT / "zh_src"
# 输出目录：必须是 docs 本身，写成 docs/zh-bank 会多一层目录，
# 导致 infer_acl 只看到 "zh-bank" 而退化成 general/public
OUT = ROOT / "docs"

# 文件名里的部门/密级编码 -> 目录层级。译文文件统一用这个前缀命名，
# 避免在正文里塞 frontmatter（PDF/PPTX 不保留 frontmatter）。
ACL_SUFFIX = "{dept}__{sens}__"


def parse_acl(stem: str) -> tuple[str, str, str]:
    """从 `<dept>__<sens>__<name>` 形式的文件名里解析出目录和真实 stem。"""
    m = re.match(r"^([a-z]+)__([a-z]+)__(.+)$", stem)
    if not m:
        return "general", "public", stem
    return m.group(1), m.group(2), m.group(3)


def count_table_rows(text: str) -> int:
    """统计表格数据行数（不含分隔行），用于校验是否丢行。"""
    rows = 0
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("|") and not re.match(r"^\|[\s:|-]+\|$", line):
            rows += 1
    return rows


# ------------------------- 生成各格式 -------------------------

def to_docx(text: str, out_path: Path) -> None:
    """Markdown -> DOCX：标题按层级，表格用 Table Grid。"""
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Noto Sans CJK SC"
    style.font.size = Pt(10.5)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")

    lines = text.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = []
            for row in block:
                if re.match(r"^\|[\s:|-]+\|$", row):
                    continue
                rows.append([c.strip() for c in row.strip("|").split("|")])
            if rows:
                ncols = max(len(r) for r in rows)
                table = doc.add_table(rows=0, cols=ncols)
                table.style = "Table Grid"
                for r in rows:
                    cells = table.add_row().cells
                    for j, val in enumerate(r + [""] * (ncols - len(r))):
                        cells[j].text = val
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = min(len(m.group(1)), 6)
            title = m.group(2).strip()
            if title:
                doc.add_heading(title, level=level)
            i += 1
            continue

        if re.match(r"^-{3,}$", stripped):
            i += 1
            continue

        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            doc.add_paragraph(m.group(1).strip(), style="List Bullet")
            i += 1
            continue

        if stripped:
            doc.add_paragraph(stripped)
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))


def to_html(text: str, out_path: Path, title: str) -> None:
    """Markdown -> HTML：表格用 <table>，标题用 h1-h6。"""
    import html as html_lib

    def inline(s: str) -> str:
        """转义 HTML 特殊字符，并把 **粗体** / *斜体* 转成对应标签。"""
        s = html_lib.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
        return s

    body: list[str] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = []
            for row in block:
                if re.match(r"^\|[\s:|-]+\|$", row):
                    continue
                rows.append([c.strip() for c in row.strip("|").split("|")])
            if rows:
                body.append("<table>")
                for idx, r in enumerate(rows):
                    tag = "th" if idx == 0 else "td"
                    body.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in r) + "</tr>")
                body.append("</table>")
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            body.append(f"<h{level}>{inline(m.group(2).strip())}</h{level}>")
            i += 1
            continue

        if re.match(r"^-{3,}$", stripped):
            i += 1
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            items = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(f"<li>{inline(lines[i].strip()[2:])}</li>")
                i += 1
            body.append("<ul>" + "".join(items) + "</ul>")
            continue

        if stripped:
            body.append(f"<p>{inline(stripped)}</p>")
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "<!DOCTYPE html>\n<html lang=\"zh-CN\">\n<head><meta charset=\"utf-8\">"
        f"<title>{html_lib.escape(title)}</title></head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n",
        encoding="utf-8",
    )


def to_pdf(text: str, out_path: Path, title: str) -> None:
    """Markdown -> PDF：reportlab + STSong-Light CID 字体，保证中文可被提取。

    不用 LibreOffice（沙箱内加载文件会失败），也不用 OnlyOffice 的 CLI
    （桌面版没有稳定的无头转换接口）；reportlab 是纯 Python，结果可预期。
    """
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    font = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(font))

    title_style = ParagraphStyle("T", fontName=font, fontSize=16, leading=22, spaceAfter=10, alignment=TA_LEFT)
    h_style = ParagraphStyle("H", fontName=font, fontSize=12, leading=17, spaceBefore=8, spaceAfter=5)
    p_style = ParagraphStyle("P", fontName=font, fontSize=9.5, leading=14, spaceAfter=4, leftIndent=8)
    cell_style = ParagraphStyle("C", fontName=font, fontSize=8.5, leading=12)

    def para(s: str, style=p_style, bullet: bool = False) -> Paragraph:
        """生成一个中文段落，剥掉 Markdown 标记；bullet=True 时加项目符号圆点。"""
        # 去掉 Markdown 标记，只保留纯文本（PDF 不需要标记）
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        s = re.sub(r"^\s*[-*]\s+", "", s)
        # 用 bulletText 画圆点，避免依赖字体里的 "·" 字形（STSong 缺这个字形）
        return Paragraph(escape(s.strip()), style, bulletText="•" if bullet else None)

    story: list = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = []
            for row in block:
                if re.match(r"^\|[\s:|-]+\|$", row):
                    continue
                rows.append([c.strip() for c in row.strip("|").split("|")])
            if rows:
                ncols = max(len(r) for r in rows)
                data = [[para(c, cell_style) for c in (r + [""] * (ncols - len(r)))] for r in rows]
                table = Table(data, repeatRows=1)
                table.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f8")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.extend([table, Spacer(1, 5)])
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            story.append(para(m.group(2).strip(), title_style if level == 1 else h_style))
            i += 1
            continue

        if re.match(r"^-{3,}$", stripped):
            i += 1
            continue

        if stripped:
            is_bullet = bool(re.match(r"^[-*]\s+", stripped))
            story.append(para(stripped, bullet=is_bullet))
        i += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm,
        title=title, author="中兴银行",
    )
    doc.build(story)


def to_pptx(text: str, out_path: Path, title: str) -> None:
    """Markdown -> PPTX：按二级标题分页，表格单独成页并自动分片。"""
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    blank = prs.slide_layouts[6]

    def add_slide(heading: str, bullets: list[str] | None = None, table_rows: list[list[str]] | None = None):
        """新建一页空白版式幻灯片：有表格就画表格，否则画项目符号列表。"""
        slide = prs.slides.add_slide(blank)
        tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.8))
        tb.text_frame.text = heading
        if table_rows:
            ncols = max(len(r) for r in table_rows)
            nrows = len(table_rows)
            gt = slide.shapes.add_table(
                nrows, ncols, Inches(0.5), Inches(1.3), Inches(9), Inches(0.4 * nrows)
            ).table
            for ri, r in enumerate(table_rows):
                for ci in range(ncols):
                    gt.cell(ri, ci).text = r[ci] if ci < len(r) else ""
        elif bullets:
            body = slide.shapes.add_textbox(Inches(0.5), Inches(1.3), Inches(9), Inches(4.5)).text_frame
            body.word_wrap = True
            first = True
            for b in bullets:
                p = body.paragraphs[0] if first else body.add_paragraph()
                first = False
                p.text = b

    lines = text.split("\n")
    i = 0
    current_heading = title
    bullets: list[str] = []
    table_rows: list[list[str]] = []

    def flush():
        """把累积的表格行或列表项落成一页幻灯片，然后清空缓冲区。"""
        nonlocal bullets, table_rows
        if table_rows:
            add_slide(current_heading, table_rows=table_rows)
        elif bullets:
            add_slide(current_heading, bullets=bullets)
        bullets, table_rows = [], []

    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            rows = []
            for row in block:
                if re.match(r"^\|[\s:|-]+\|$", row):
                    continue
                rows.append([c.strip() for c in row.strip("|").split("|")])
            if rows:
                flush()
                chunk = []
                for r in rows:
                    chunk.append(r)
                    # 表格超过 12 行就换页，避免溢出幻灯片
                    if len(chunk) >= 12:
                        table_rows = chunk
                        flush()
                        chunk = []
                if chunk:
                    table_rows = chunk
                    flush()
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            flush()
            current_heading = m.group(2).strip() or title
            i += 1
            continue

        if re.match(r"^-{3,}$", stripped):
            i += 1
            continue

        if stripped:
            s = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
            s = re.sub(r"^\s*[-*]\s+", "", s)
            bullets.append(s)
        i += 1

    flush()
    if len(prs.slides) == 0:
        add_slide(title, bullets=[text[:200]])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))


def build_formats(md_text: str, stem: str, out_dir: Path) -> dict[str, Path]:
    """由一份中文 Markdown 生成 5 种格式，返回 {格式: 路径}。"""
    title = md_text.lstrip("# ").split("\n")[0].strip()[:80] or stem
    paths: dict[str, Path] = {}

    p = out_dir / f"{stem}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md_text + "\n", encoding="utf-8")
    paths["md"] = p

    to_docx(md_text, out_dir / f"{stem}.docx")
    paths["docx"] = out_dir / f"{stem}.docx"

    to_pdf(md_text, out_dir / f"{stem}.pdf", title)
    paths["pdf"] = out_dir / f"{stem}.pdf"

    to_html(md_text, out_dir / f"{stem}.html", title)
    paths["html"] = out_dir / f"{stem}.html"

    to_pptx(md_text, out_dir / f"{stem}.pptx", title)
    paths["pptx"] = out_dir / f"{stem}.pptx"

    return paths


# ------------------------- 校验 -------------------------

def verify_with_loader(md_text: str, out_dir: Path, stem: str) -> list[str]:
    """用项目自己的 loader 回读每个格式，确认能解析且内容没有大量丢失。"""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from rag_learning.loaders import load_document  # noqa: E402

    issues: list[str] = []

    def substantial(text: str) -> str:
        """去掉空白与所有 Markdown 标记，只留实质文字。

        PDF/PPTX/HTML 不保留 | - : * # 这些排版标记，按原始字符数比较会把
        "标记消失"误判成"内容丢失"，所以两边都先归一化再比。
        """
        return re.sub(r"[\s|\-:*#>_`~\[\]()]", "", text)

    # 以"实质片段覆盖率"为准：把 md 切成句子，看有多少能在目标格式里找到。
    # 这比字符数更准确，能区分"标记丢失"和"内容真的缺了"。
    segments = [substantial(s) for s in re.split(r"[。\n]", md_text)]
    segments = [s for s in segments if len(s) >= 8]
    base_len = len(substantial(md_text))

    for ext in ("md", "docx", "pdf", "html", "pptx"):
        p = out_dir / f"{stem}.{ext}"
        if not p.exists():
            issues.append(f"{ext}: 文件未生成")
            continue
        try:
            doc = load_document(p, out_dir)
            body = substantial(doc.text)
            n = len(body)
            ratio = n / base_len if base_len else 0
            found = sum(1 for seg in segments if seg in body)
            coverage = found / len(segments) if segments else 1.0
            # 以片段覆盖率为主判据（>=95% 通过），字符数仅作参考
            flag = "" if coverage >= 0.95 else "  <-- 有内容缺失"
            print(f"    {ext:5s} OK  {n:6d} 字  片段覆盖 {coverage:.0%}  (字符 {ratio:.0%}){flag}")
            if coverage < 0.95:
                issues.append(f"{ext}: 片段覆盖 {coverage:.0%} ({found}/{len(segments)})")
        except Exception as exc:  # noqa: BLE001
            issues.append(f"{ext}: 解析失败 {type(exc).__name__}: {exc}")
    return issues


def main() -> int:
    """命令行入口：遍历源目录，生成各格式并回读校验，全部通过返回 0。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只处理指定 stem")
    ap.add_argument("--verify", action="store_true", help="只校验，不重新生成")
    ap.add_argument("--src", default=str(SRC), help="中文 md 源目录")
    ap.add_argument("--out", default=str(OUT), help="输出目录")
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_root = Path(args.out)
    if not src_dir.exists():
        print(f"源目录不存在：{src_dir}")
        print("请先把中文 Markdown 放到该目录，文件名格式 <部门>__<密级>__<name>.md")
        return 1

    files = sorted(src_dir.glob("*.md"))
    if args.only:
        wanted = set(args.only)
        files = [f for f in files if any(w in f.stem for w in wanted)]
    if not files:
        print("没有找到待处理文件")
        return 1

    print(f"源: {src_dir}\n输出: {out_root}\n待处理: {len(files)} 个\n")
    ok = fail = 0
    for idx, src in enumerate(files, start=1):
        dept, sens, name = parse_acl(src.stem)
        doc_dir = out_root / dept / sens
        md_text = src.read_text(encoding="utf-8").strip()
        print(f"[{idx}/{len(files)}] {src.name} -> {dept}/{sens}/{name}  ({len(md_text)} 字)")

        if not args.verify:
            build_formats(md_text, name, doc_dir)
        issues = verify_with_loader(md_text, doc_dir, name)
        if issues:
            print("    问题:")
            for it in issues:
                print(f"      - {it}")
            fail += 1
        else:
            print("    通过")
            ok += 1

    print(f"\n完成: 成功 {ok}, 失败 {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
