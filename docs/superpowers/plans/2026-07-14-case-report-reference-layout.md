# 个例报告参考版式实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将个例多维检索 PDF 改为与 `resource/FST2025-6.pdf` 一致的简洁文字分析报告，并生成真实报告供用户验收。

**Architecture:** 在 `case_multidim_search` 目录新增连续排版组件，统一管理 A4 页面、参考模板字号、段落流、分页、图片和图注。`PdfReportBuilder` 只负责把现有分析结果组织成“综合概况、主要分析、代表个例、结论建议、数据局限”，不修改检索、分析、图片重定位和接口协议。

**Tech Stack:** Python 3.10、Matplotlib `PdfPages`、Pillow、pypdf、pypdfium2、unittest。

---

## 文件结构

- 创建 `backend/app/services/agent/case_multidim_search/report_document.py`：连续排版、自动分页、标题、正文、图片、图注和页码。
- 修改 `backend/app/services/agent/case_multidim_search/pdf_report.py`：按技术报告章节组织现有分析数据。
- 修改 `backend/app/services/agent/case_multidim_search/chart_tool.py`：把统计图调整为白底、克制配色和适合正文插图的比例。
- 创建 `tests/case_multidim_search/test_reference_report_layout.py`：验证模板参数、无装饰页眉、正文优先和内部字段隐藏。
- 保留 `tests/case_multidim_search/test_report_quality.py` 等现有回归测试。

### Task 1: 用失败测试固定参考模板要求

**Files:**
- Create: `tests/case_multidim_search/test_reference_report_layout.py`
- Test: `tests/case_multidim_search/test_report_quality.py`

- [ ] **Step 1: 写模板参数失败测试**

```python
class ReferenceReportLayoutTests(unittest.TestCase):
    """验证报告与参考 PDF 的页面和文字层级一致。"""

    def test_layout_constants_match_reference_pdf(self):
        """验证页边距、正文、标题、图注和行距采用参考值。"""
        self.assertAlmostEqual(85.1 / 595.3, ReferenceReportDocument.LEFT_MARGIN, places=3)
        self.assertEqual(12.0, ReferenceReportDocument.BODY_FONT_SIZE)
        self.assertEqual(15.0, ReferenceReportDocument.SECTION_FONT_SIZE)
        self.assertAlmostEqual(14.0, ReferenceReportDocument.SUBSECTION_FONT_SIZE, places=1)
        self.assertEqual(9.0, ReferenceReportDocument.CAPTION_FONT_SIZE)
        self.assertAlmostEqual(23.4 / 841.9, ReferenceReportDocument.BODY_LINE_STEP, places=3)
```

- [ ] **Step 2: 写无装饰页眉失败测试**

测试生成一份最小报告并渲染首页顶部三分之一，断言深色像素占比小于 `0.15`、页面角落为白色。旧版深色封面会使测试失败。

```python
top = rendered.crop((0, 0, rendered.width, rendered.height // 3)).convert("RGB")
dark_ratio = sum(1 for r, g, b in top.get_flattened_data() if max(r, g, b) < 80) / (top.width * top.height)
self.assertLess(dark_ratio, 0.15)
self.assertTrue(all(value >= 248 for value in rendered.getpixel((5, 5))))
```

- [ ] **Step 3: 写个例分析文本失败测试**

```python
text = PdfReportBuilder()._case_analysis(case, metrics)
self.assertIn("过程特征", text)
self.assertIn("中心点最大降水量", text)
self.assertNotIn("融合得分", text)
self.assertNotIn("向量", text)
```

- [ ] **Step 4: 运行测试确认失败原因正确**

Run: `python -m unittest tests.case_multidim_search.test_reference_report_layout -v`

Expected: FAIL，原因是 `ReferenceReportDocument` 和 `_case_analysis` 尚不存在，或旧首页深色像素比例过高。

### Task 2: 实现连续技术报告排版组件

**Files:**
- Create: `backend/app/services/agent/case_multidim_search/report_document.py`
- Test: `tests/case_multidim_search/test_reference_report_layout.py`

- [ ] **Step 1: 定义参考模板常量和中文字体**

```python
class ReferenceReportDocument:
    """按照参考技术报告参数连续编排 A4 页面。"""

    PAGE_SIZE = (8.27, 11.69)
    LEFT_MARGIN = 85.1 / 595.3
    RIGHT_MARGIN = 1 - LEFT_MARGIN
    BODY_FONT_SIZE = 12.0
    SECTION_FONT_SIZE = 15.0
    SUBSECTION_FONT_SIZE = 14.04
    CAPTION_FONT_SIZE = 9.0
    BODY_LINE_STEP = 23.4 / 841.9
```

正文优先使用 `C:/Windows/Fonts/simsun.ttc`，标题使用 `simhei.ttf`，无对应字体时回退到现有中文字体。

- [ ] **Step 2: 实现页面生命周期和居中页码**

实现 `_new_page()`、`_save_page()`、`finish()`。页面背景固定白色，正文区域从约 `y=0.895` 开始，低于页脚安全线时自动分页；页脚只绘制居中页码。

- [ ] **Step 3: 实现标题、章节和正文段落流**

实现 `add_report_title()`、`add_section()`、`add_subsection()`、`add_paragraph()`。正文按中文字符宽度换行，首行缩进两个汉字，段落之间保留小于一行的间距。

- [ ] **Step 4: 实现等比例图片和图注**

实现 `add_figure(path, caption)`：先校验图片，再按正文宽度等比例缩放；空间不足时先分页；图片下方以 `9 pt` 居中图注并递增“图 N”。不得拉伸或裁切图片。

- [ ] **Step 5: 运行模板参数测试**

Run: `python -m unittest tests.case_multidim_search.test_reference_report_layout.ReferenceReportLayoutTests.test_layout_constants_match_reference_pdf -v`

Expected: PASS。

### Task 3: 把报告内容改为连续分析正文

**Files:**
- Modify: `backend/app/services/agent/case_multidim_search/pdf_report.py`
- Test: `tests/case_multidim_search/test_reference_report_layout.py`

- [ ] **Step 1: 删除卡片式页面编排入口**

`build()` 不再调用 `_cover_page()`、`_findings_page()`、`_charts_page()`、固定 `_case_page()` 和彩色 `_conclusion_page()`，改为创建 `ReferenceReportDocument` 并按章节连续写入。

- [ ] **Step 2: 编排首页与综合概况**

首页顺序固定为报告标题、生成日期、“综合概况”、执行摘要和重点发现自然段。重点发现使用“（1）…（2）…”的完整句式，不绘制编号卡片。

- [ ] **Step 3: 编排主要分析并内嵌统计图**

建立章节映射：

```python
sections = [
    ("1. 时间分布特征", analysis["sections"]["temporal"]),
    ("2. 灾种结构特征", analysis["sections"]["disaster"]),
    ("3. 空间影响特征", analysis["sections"]["spatial"]),
    ("4. 中心点强度特征", analysis["sections"]["intensity"]),
    ("5. 证据完整度", analysis["sections"]["evidence"]),
]
```

每个分析段落之后最多插入一张含义对应的统计图，不再创建独立图表页。

- [ ] **Step 4: 编排代表个例分析**

实现 `_case_analysis(case, metrics)`，输出过程特征、中心点强度及业务研判，不包含融合得分或向量检索字段。每个个例正文之后插入最多两张真实数据库图片。

- [ ] **Step 5: 编排结论、建议与数据局限**

结论使用完整自然段；建议采用“1. …”编号文字；局限使用普通段落，不绘制彩色警示框。

- [ ] **Step 6: 运行新测试并确认通过**

Run: `python -m unittest tests.case_multidim_search.test_reference_report_layout -v`

Expected: PASS。

### Task 4: 简化正文统计图风格

**Files:**
- Modify: `backend/app/services/agent/case_multidim_search/chart_tool.py`
- Test: `tests/case_multidim_search/test_reporting_tools.py`

- [ ] **Step 1: 写统计图白底和正文比例测试**

验证输出图为白底、宽高比约 `1.8`、坐标标题保持中文。

- [ ] **Step 2: 调整统计图渲染参数**

将画布调整为约 `(8.0, 4.4)`，使用深灰主色和单一强调色，去除彩色分类轮换；保留数值标签、中文坐标和必要网格。

- [ ] **Step 3: 运行绘图回归测试**

Run: `python -m unittest tests.case_multidim_search.test_reporting_tools tests.case_multidim_search.test_report_refinement -v`

Expected: PASS。

### Task 5: 全量回归并生成真实验收报告

**Files:**
- Verify: `backend/app/services/agent/case_multidim_search`
- Verify: `tests/case_multidim_search`
- Generate: `data/case_multidim_reports/postman-smoke-report.pdf`

- [ ] **Step 1: 运行专项和 Agent 回归测试**

Run:

```powershell
python -m unittest discover -s tests/case_multidim_search -v
python -m unittest tests.test_agent_intent tests.test_agent_planner tests.test_agent_answer_synthesizer -v
python -m compileall -q backend/app/services/agent/case_multidim_search tests/case_multidim_search
```

Expected: 全部 PASS，无语法错误。

- [ ] **Step 2: 检查全部方法中文注释**

使用 AST 扫描两个目标目录，要求所有 `FunctionDef` 和 `AsyncFunctionDef` 都有包含中文字符的 docstring，`ISSUES=0`。

- [ ] **Step 3: 通过 FastAPI 报告路由生成真实 PDF**

使用当前真实标准个例库和图片元数据请求 `/api/case-multidim/report`，覆盖生成 `data/case_multidim_reports/postman-smoke-report.pdf`。

- [ ] **Step 4: 逐页渲染验收**

用 pypdfium2 检查：所有页面为约 `595.4 x 841.7 pt`、页面非空、无大面积深色页眉、首页角落为白色、图片像素真实存在。将首页、正文分析页、代表个例图文页和结论页渲染为 PNG 进行视觉复核。

- [ ] **Step 5: 向用户提供最终报告**

返回最终 PDF 的绝对路径和测试结果，并明确说明浏览器或本地预览中看到的是本次重新生成的文件。
