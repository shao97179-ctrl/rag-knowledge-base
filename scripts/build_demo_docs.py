"""【已废弃】最早的 10 篇演示文档生成脚本。

docs/ 下的语料已替换为「中兴银行」中文语料（由 scripts/build_zh_corpus.py 生成），
本脚本原先写入 docs/ 的英文演示文档已被移除，保留代码仅供追溯。

如需重新生成演示文档，请把输出目录改到 docs 之外再运行，避免污染正式语料。
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from openpyxl import Workbook
from openpyxl.styles import Font
from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs_legacy_demo"  # 独立目录，绝不写入正式语料


def write_text(rel_path: str, content: str) -> Path:
    """把文本内容写到 docs/<rel_path>，顺手建好父目录，返回写入路径。"""
    path = DOCS / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _setup_docx_font(document: Document) -> None:
    """设置 DOCX 的默认中文字体，否则解析出的中文可能缺字形。"""
    style = document.styles["Normal"]
    style.font.name = "Noto Sans CJK SC"
    style.font.size = Pt(10.5)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Noto Sans CJK SC")


def _add_table(document: Document, header: list[str], rows: list[list[str]]) -> None:
    """按表头列数建一张 Table Grid 样式的表，先写表头再逐行写数据。"""
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    for index, text in enumerate(header):
        table.rows[0].cells[index].text = text
    for row in rows:
        cells = table.add_row().cells
        for index, text in enumerate(row):
            cells[index].text = text


# ---------------------------------------------------------------------------
# 1. Markdown：员工手册（frontmatter 声明权限）
# ---------------------------------------------------------------------------

def build_handbook() -> Path:
    """生成员工手册（Markdown，frontmatter 声明 general/public）。"""
    return write_text(
        "general/public/company-handbook.md",
        """
---
title: 员工手册（2026 版）
department: general
sensitivity: public
---

# 员工手册（2026 版）

## 公司简介

星云科技成立于 2018 年，主营企业级数据平台与运维服务，总部位于杭州，在北京、深圳设有研发中心。公司现有员工约 1200 人，其中研发与运维人员占比超过 60%。

## 办公与考勤

标准工作时间为周一至周五 9:30-18:30，午休 12:00-13:30。核心协作时间为 10:30-17:00，会议尽量安排在这个区间。

| 项目 | 规定 |
| --- | --- |
| 弹性上班 | 9:30-10:30 之间到岗，满 8 小时即可 |
| 远程办公 | 每周最多 2 天，需直属主管提前审批 |
| 加班调休 | 工作日加班按 1:1 调休，法定节假日按国家规定 |
| 考勤异常 | 当月累计 3 次未打卡需提交说明 |

## 信息安全基本要求

- 公司账号必须开启二次验证，密码至少 12 位并每 90 天更换一次。
- 机密资料不得通过个人网盘、私人邮箱或未审批的第三方工具传输。
- 笔记本电脑必须开启磁盘加密，离开工位时锁屏。
- 发现疑似数据泄露，立即联系 security@example.com 或拨打 400-000-1234。

## 常用入口

报销、请假、采购分别在公司门户的财务、人事、行政模块发起。IT 报修通过服务台提交，紧急故障直接联系值班工程师。
""",
    )


# ---------------------------------------------------------------------------
# 2. CSV：供应商清单
# ---------------------------------------------------------------------------

def build_vendors() -> Path:
    """生成供应商清单（CSV，internal），用来演示表格解析。"""
    return write_text(
        "general/internal/vendor-list.csv",
        """
供应商名称,类别,联系人,服务级别,合同到期日,续签负责人,备注
云杉数据,数据库与中间件,王敏,A,2026-12-31,采购部-李强,到期前 90 天启动续签评估
星河云,云资源,陈涛,A,2027-03-31,采购部-李强,年度框架协议
蓝盾安全,安全服务,赵磊,B,2026-09-30,信息安全部-周宁,含渗透测试与应急响应
恒信咨询,管理咨询,孙悦,C,2026-06-30,行政部-吴倩,按项目结算
迅达物流,办公物流,刘洋,C,2026-11-15,行政部-吴倩,仅覆盖杭州总部
""",
    )


# ---------------------------------------------------------------------------
# 3. DOCX：请假与考勤制度
# ---------------------------------------------------------------------------

def build_leave_policy() -> Path:
    """生成请假制度 DOCX（hr/internal），带多级标题和一张审批表格。"""
    path = DOCS / "hr/internal/leave-policy.docx"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    _setup_docx_font(document)
    document.add_heading("请假与考勤制度", level=1)
    document.add_paragraph("本制度适用于公司全体正式员工，试用期员工参照执行。")

    document.add_heading("一、年假", level=2)
    document.add_paragraph("入职满 1 年享受 5 天年假，满 3 年 10 天，满 5 年 15 天，满 10 年 20 天。")
    document.add_paragraph("年假可以跨年顺延，但最多保留到次年 3 月 31 日；逾期未休部分视为放弃，法定年假除外。")
    document.add_paragraph("申请年假需提前 3 个工作日在公司门户提交，连续休假超过 5 天需部门负责人审批。")

    document.add_heading("二、病假", level=2)
    document.add_paragraph("病假需提供二级以上医院出具的证明。病假期间工资按当地最低工资标准的 80% 发放，具体以劳动合同为准。")
    document.add_paragraph("连续病假超过 3 天，需同步抄送 HRBP；超过 15 天按医疗期规定处理。")

    document.add_heading("三、事假", level=2)
    document.add_paragraph("事假为无薪假，需提前 1 个工作日申请。突发情况可先电话报备，事后 24 小时内补提流程。")

    document.add_heading("四、调休", level=2)
    document.add_paragraph("工作日加班按 1:1 调休，调休需在加班发生后 6 个月内使用。法定节假日加班按国家规定支付加班费。")

    document.add_heading("五、请假流程与审批", level=2)
    _add_table(
        document,
        ["假期类型", "是否需要证明", "审批人", "工资处理"],
        [
            ["年假", "否", "直属主管；超过 5 天加部门负责人", "正常发放"],
            ["病假", "二级以上医院证明", "直属主管 + HRBP", "按当地最低工资 80%"],
            ["事假", "否", "直属主管", "无薪"],
            ["调休", "加班记录", "直属主管", "正常发放"],
        ],
    )
    document.add_paragraph("3 天以内由直属主管审批，3 天以上由部门负责人审批，5 天以上需 HRBP 备案。")
    document.save(str(path))
    return path


# ---------------------------------------------------------------------------
# 4. XLSX：薪酬带宽（机密）
# ---------------------------------------------------------------------------

def build_salary_band() -> Path:
    """生成薪酬带宽 XLSX（hr/confidential），三个工作表用来演示多 sheet 解析。"""
    path = DOCS / "hr/confidential/salary-band.xlsx"
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "2026薪酬带宽"
    sheet.append(["职级", "月薪下限", "月薪上限", "年度奖金月数", "股权授予(股)", "备注"])
    rows = [
        ["P4", 18000, 24000, 2, 0, "应届/初级"],
        ["P5", 24000, 32000, 3, 500, "骨干"],
        ["P6", 32000, 45000, 4, 1500, "资深"],
        ["P7", 45000, 62000, 5, 4000, "专家"],
        ["P8", 62000, 85000, 6, 10000, "架构师/经理"],
    ]
    for row in rows:
        sheet.append(row)

    bonus = workbook.create_sheet("奖金系数")
    bonus.append(["部门", "系数下限", "系数上限", "说明"])
    for row in [
        ["研发", 0.8, 1.5, "按项目交付质量"],
        ["运维", 0.8, 1.3, "按稳定性指标"],
        ["销售", 0.5, 2.0, "按回款完成率"],
        ["职能", 0.8, 1.2, "按年度评级"],
    ]:
        bonus.append(row)

    equity = workbook.create_sheet("股权激励")
    equity.append(["批次", "授予时间", "归属周期", "备注"])
    for row in [
        ["2025A", "2025-04-01", "4 年按 25% 归属", "离职未归属部分收回"],
        ["2026A", "2026-04-01", "4 年按 25% 归属", "待董事会批准"],
    ]:
        equity.append(row)

    for sheet in workbook.worksheets:
        for cell in sheet[1]:
            cell.font = Font(bold=True)
        for column in sheet.columns:
            letter = column[0].column_letter
            sheet.column_dimensions[letter].width = 18
    workbook.save(str(path))
    return path


# ---------------------------------------------------------------------------
# 5. PDF：报销流程（docx -> pdf）
# ---------------------------------------------------------------------------

def build_expense_pdf() -> Path:
    """用 reportlab 直接生成中文 PDF；LibreOffice 在沙箱里无法加载文件。"""
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = DOCS / "finance/internal/expense-reimbursement.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    font_name = "STSong-Light"
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    title_style = ParagraphStyle("CnTitle", fontName=font_name, fontSize=20, leading=26, spaceAfter=12)
    heading_style = ParagraphStyle("CnHeading", fontName=font_name, fontSize=14, leading=20, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("CnBody", fontName=font_name, fontSize=10.5, leading=17, spaceAfter=6)
    cell_style = ParagraphStyle("CnCell", fontName=font_name, fontSize=9.5, leading=13)

    def para(text: str, style=body_style) -> Paragraph:
        """生成一个中文段落；正文要先 escape，避免 & 和 <> 被当成标记。"""
        return Paragraph(escape(text), style)

    story: list = [para("费用报销管理办法", title_style)]
    story.append(para("本办法适用于全体员工因公发生的差旅、交通、招待、办公采购等费用。"))

    story.append(para("一、报销时限", heading_style))
    story.append(para("差旅费用需在出差结束后 10 个工作日内提交；跨月费用最迟在当月 28 日前提交，逾期需部门负责人说明原因。"))

    story.append(para("二、发票要求", heading_style))
    story.append(para("发票抬头必须为“星云科技有限公司”，税号 91330100MA2XXXXXXX。"))
    story.append(para("增值税专用发票需附行程单；电子发票需上传 PDF 原件，截图和照片不予受理。"))
    story.append(para("发票内容应与实际业务一致，连号发票、模糊发票、个人消费发票会被退回。"))

    story.append(para("三、差旅标准", heading_style))
    table_data = [
        ["城市级别", "住宿上限(元/晚)", "市内交通", "餐补(元/天)"],
        ["一线城市", "600", "实报实销", "100"],
        ["新一线城市", "450", "实报实销", "80"],
        ["其他城市", "350", "实报实销", "60"],
    ]
    table = Table(
        [[para(cell, cell_style) for cell in row] for row in table_data],
        colWidths=[45 * mm, 45 * mm, 45 * mm, 40 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f8")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([table, Spacer(1, 6)])
    story.append(para("超标部分需事前审批，并在报销单中附审批截图，否则按标准上限报销。"))

    story.append(para("四、审批流程", heading_style))
    story.append(para("直属主管 → 部门负责人 → 财务复核。单笔超过 5000 元需加签 CFO；招待费超过 2000 元需附事由和参与人名单。"))

    story.append(para("五、常见退回原因", heading_style))
    story.append(para("发票抬头错误、缺少行程单、超标准未说明、报销单金额与发票不一致、重复报销。被退回的单据需在 3 个工作日内修正。"))

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="费用报销管理办法",
        author="星云科技",
    )
    document.build(story)
    return output


# ---------------------------------------------------------------------------
# 6. HTML：故障响应手册
# ---------------------------------------------------------------------------

def build_incident_html() -> Path:
    """生成故障响应手册 HTML（ops/public），含表格、有序和无序列表。"""
    return write_text(
        "ops/public/incident-response.html",
        """
<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>故障响应手册</title></head>
<body>
  <h1>故障响应手册</h1>
  <p>本手册适用于生产环境故障的分级、响应、升级与复盘。所有值班人员必须熟悉。</p>

  <h2>故障分级</h2>
  <table>
    <tr><th>级别</th><th>定义</th><th>响应时限</th><th>初判时限</th></tr>
    <tr><td>P0</td><td>核心交易不可用、数据丢失风险</td><td>5 分钟</td><td>30 分钟</td></tr>
    <tr><td>P1</td><td>主要功能降级或大面积超时</td><td>15 分钟</td><td>1 小时</td></tr>
    <tr><td>P2</td><td>非核心功能异常</td><td>1 小时</td><td>次工作日</td></tr>
    <tr><td>P3</td><td>咨询、优化建议</td><td>次工作日</td><td>次工作日</td></tr>
  </table>

  <h2>响应流程</h2>
  <ol>
    <li>值班工程师在告警群确认故障，5 分钟内响应并认领。</li>
    <li>P0/P1 立即拉起临时作战群，通知值班经理和相关研发负责人。</li>
    <li>先止损再定位：必要时摘流量、限流、回滚最近一次变更。</li>
    <li>每 30 分钟同步一次进展，直到故障关闭。</li>
  </ol>
  <p>数据库连接池耗尽、慢查询导致的核心接口超时，通常按 P1 处理；如果影响支付主链路则升级为 P0。</p>

  <h2>升级路径</h2>
  <p>值班工程师 → 值班经理 → 技术负责人 → CTO。P0 故障需同步通知客户成功团队，由其在 30 分钟内通知受影响客户。</p>

  <h2>值班与复盘</h2>
  <ul>
    <li>值班电话：400-000-1234；值班群：星云-oncall。</li>
    <li>P0/P1 故障必须在 3 个工作日内输出复盘报告，包含时间线、根因、改进项和责任人。</li>
    <li>复盘报告归档到运维知识库，并在下一次值班交接中宣讲。</li>
  </ul>
</body>
</html>
""",
    )


# ---------------------------------------------------------------------------
# 7. TXT：数据库连接池排查手册
# ---------------------------------------------------------------------------

def build_connection_pool() -> Path:
    """生成连接池排查手册 TXT（ops/internal），金牌检索用例的主要来源。"""
    return write_text(
        "ops/internal/db-connection-pool.txt",
        """
# 数据库连接池耗尽排查手册

## 现象

应用日志持续出现 timeout acquiring connection from pool、HikariPool-1 - Connection is not available 等错误；接口 P99 延迟升高，部分请求直接返回 500。数据库监控显示活跃连接数接近 max_connections。

## 一分钟判断

1. 执行 SELECT state, count(*) FROM pg_stat_activity GROUP BY state; 观察 active、idle、idle in transaction 的数量。
2. 查看应用侧连接池指标：active、idle、waiting、maximumPoolSize。
3. 对比数据库 max_connections、pgbouncer default_pool_size 与所有实例的连接池上限之和。
4. 检查最近 30 分钟是否有发布、扩容、慢查询或流量突增。

## 应急处理

1. 先在告警群同步现象和影响范围，避免多人重复操作。
2. 如果是单个实例连接泄漏，先摘除该实例流量，再滚动重启。
3. 对长期 idle in transaction 的连接，确认业务影响后谨慎终止；不要一次性杀掉全部连接。
4. 如果是突发流量，临时调大 pgbouncer 的 default_pool_size，并设置合理的 query_wait_timeout。
5. 如果伴随慢查询，先 kill 掉执行超过 60 秒的查询，再保留现场。

## 根因排查

- 连接泄漏：检查代码中 Connection、Statement、ResultSet 是否在 finally 中关闭；重点看新上线的定时任务和异步线程。
- 慢查询：用 pg_stat_statements 按 total_exec_time 排序，确认是否有缺失索引或全表扫描。
- 连接池配置：所有应用实例的 maximumPoolSize 之和不应超过数据库 max_connections 的 80%，要给运维和监控留余量。
- 事务过长：检查是否有事务中包含 HTTP 调用、文件上传等耗时操作。
- 突发流量：结合网关 QPS 和线程池指标判断，必要时限流。

## 修复与复盘

修复后持续观察 30 分钟连接曲线，确认 waiting 归零。P1 及以上故障需要输出复盘，补充连接池容量基线、告警阈值和演练计划。
""",
    )


# ---------------------------------------------------------------------------
# 8. Markdown：值班手册
# ---------------------------------------------------------------------------

def build_oncall_runbook() -> Path:
    """生成运维值班手册 Markdown（ops/internal），与连接池手册互为干扰项。"""
    return write_text(
        "ops/internal/oncall-runbook.md",
        """
# 运维值班手册

## 交接班要求

- 交接时间为每天 09:30 和 21:30，交接双方需在值班群确认。
- 必须记录：未闭环告警、进行中的变更、已知风险、待跟进事项。
- 交接完成后，接班人负责更新值班看板。

## 常见告警处理

### 数据库连接池使用率超过 85%

先确认是否为业务高峰，再检查是否有慢查询或连接泄漏。具体步骤参考《数据库连接池耗尽排查手册》。如果 10 分钟内无法定位，按 P1 升级。

### API 5xx 比例超过 1%

1. 查看网关错误分布，确认是单接口还是全站。
2. 检查最近一次发布和配置变更，优先回滚。
3. 如果伴随数据库超时，转到连接池排查流程。

### 磁盘使用率超过 80%

先清理日志和临时文件，再评估扩容。不要直接删除业务数据目录。数据库磁盘超过 85% 需要立即通知 DBA。

## 重启与回滚

- 重启前先摘流量，确认没有进行中的批处理任务。
- 滚动重启，单次只操作一个实例，观察 5 分钟再继续。
- 回滚条件：错误率持续 5 分钟高于 5%，或核心接口 P99 超过基线 2 倍。
- 回滚后保留现场日志，不要立即清理。

## 升级路径

- P0/P1：立即电话值班经理，同时在作战群 @ 技术负责人。
- P2：在值班群登记，按排期处理。
- P3：转交对应研发团队，值班人员只做记录。
""",
    )


# ---------------------------------------------------------------------------
# 9. PPTX：API 设计规范
# ---------------------------------------------------------------------------

def build_api_slides() -> Path:
    """生成 API 设计规范 PPTX（rd/internal），同时覆盖文本框、表格和备注。"""
    path = DOCS / "rd/internal/api-guidelines.pptx"
    path.parent.mkdir(parents=True, exist_ok=True)
    presentation = Presentation()

    slide = presentation.slides.add_slide(presentation.slide_layouts[0])
    slide.shapes.title.text = "API 设计规范"
    slide.placeholders[1].text = "研发中心 · 2026 版"

    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "命名与版本"
    frame = slide.placeholders[1].text_frame
    frame.text = "URL 使用小写 kebab-case，例如 /api/v1/order-items"
    for line in [
        "版本号放在路径中：/api/v1、/api/v2",
        "资源名使用复数，避免动词：GET /users 而不是 /getUsers",
        "查询参数使用 snake_case，保持跨语言一致",
        "废弃接口至少保留 6 个月，并在响应头返回 Sunset",
    ]:
        paragraph = frame.add_paragraph()
        paragraph.text = line

    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "统一错误码"
    rows = [
        ["错误码", "HTTP 状态", "含义", "处理建议"],
        ["40001", "400", "参数校验失败", "返回具体字段"],
        ["40101", "401", "未认证或 token 过期", "刷新 token"],
        ["40301", "403", "无权限", "检查 scope"],
        ["42901", "429", "触发限流", "指数退避重试"],
        ["50001", "500", "服务内部错误", "记录 trace_id"],
    ]
    table = slide.shapes.add_table(len(rows), 4, Inches(0.5), Inches(1.6), Inches(9), Inches(3)).table
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            table.cell(row_index, column_index).text = value

    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "分页与过滤"
    frame = slide.placeholders[1].text_frame
    frame.text = "列表接口默认分页，page_size 最大 100"
    for line in [
        "偏移分页：page + page_size，适合后台管理页面",
        "游标分页：cursor + limit，适合大数据量和无限滚动",
        "过滤参数必须显式声明，禁止把任意字段拼进 SQL",
        "排序参数使用 sort=field:asc 的形式，只允许白名单字段",
    ]:
        paragraph = frame.add_paragraph()
        paragraph.text = line

    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "鉴权与幂等"
    frame = slide.placeholders[1].text_frame
    frame.text = "使用 OAuth2 + JWT，access token 有效期 2 小时"
    for line in [
        "refresh token 有效期 30 天，支持主动吊销",
        "按 scope 做最小权限控制，禁止万能 token",
        "POST 接口支持 Idempotency-Key，防止重复下单",
        "重试使用指数退避，最多 3 次；只对幂等接口自动重试",
    ]:
        paragraph = frame.add_paragraph()
        paragraph.text = line

    presentation.save(str(path))
    return path


# ---------------------------------------------------------------------------
# 10. DOCX：架构评审纪要（机密）
# ---------------------------------------------------------------------------

def build_architecture_review() -> Path:
    """生成架构评审纪要 DOCX（rd/confidential），用于演示最高密级权限。"""
    path = DOCS / "rd/confidential/architecture-review.docx"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = Document()
    _setup_docx_font(document)
    document.add_heading("架构评审纪要（机密）", level=1)
    document.add_paragraph("会议时间：2026-09-18。参与人：架构组、DBA、运维负责人、研发经理。")

    document.add_heading("一、背景", level=2)
    document.add_paragraph("现网使用 PostgreSQL 14 单主实例，峰值 QPS 约 3200，连接数峰值 850。按业务增长预测，明年峰值将增长 2.2 倍，现有实例和连接池配置无法支撑。")

    document.add_heading("二、数据库选型对比", level=2)
    _add_table(
        document,
        ["方案", "优点", "缺点", "结论"],
        [
            ["PostgreSQL 16 主从", "迁移成本低，团队熟悉", "写扩展有限", "首选"],
            ["MySQL 8 分库分表", "写扩展好", "改造量大，事务语义变化", "备选"],
            ["自研分库中间件", "灵活", "维护成本高", "暂不采用"],
        ],
    )

    document.add_heading("三、容量评估", level=2)
    document.add_paragraph("按 2.2 倍增长估算，明年峰值 QPS 约 7000，连接数约 1800。建议将 max_connections 调整为 1200，并引入 pgbouncer 做连接复用，应用侧连接池上限统一下调。")

    document.add_heading("四、风险与未决事项", level=2)
    document.add_paragraph("迁移窗口需要与财务月结错开；双写期间要重点关注数据一致性；连接池参数需要重新压测，避免连接池耗尽。灾备切换演练尚未排期。")

    document.add_heading("五、迁移计划", level=2)
    document.add_paragraph("Q4 完成双写和回滚预案，Q1 切换读流量，Q2 下线旧库。每个阶段都需要灰度，并保留至少 7 天可回滚窗口。")

    document.add_heading("六、决议", level=2)
    document.add_paragraph("采用 PostgreSQL 16 主从方案，由 DBA 牵头在 10 月底前完成压测报告；运维负责 pgbouncer 接入和连接池基线；研发负责双写改造。")
    document.save(str(path))
    return path


def main() -> None:
    """依次生成 10 篇演示文档，最后统计 docs 下的文件总数。"""
    builders = [
        build_handbook,
        build_vendors,
        build_leave_policy,
        build_salary_band,
        build_expense_pdf,
        build_incident_html,
        build_connection_pool,
        build_oncall_runbook,
        build_api_slides,
        build_architecture_review,
    ]
    for builder in builders:
        path = builder()
        print(f"生成：{path.relative_to(ROOT)}")
    files = sorted(path for path in DOCS.rglob("*") if path.is_file())
    print(f"\n共 {len(files)} 个文档")


if __name__ == "__main__":
    main()
