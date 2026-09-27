# 个例多维检索智能体设计

## 目标

在独立目录 `backend/app/services/agent/case_multidim_search/` 内实现可迁移的个例多维检索智能体。智能体接收自然语言，结合标准化个例结构化过滤和文档向量检索，返回可追溯结果，并导出包含统计图和原始证据图的中文 PDF 报告。

## 边界

- 所有智能体生产代码、路由和报告组件都位于同一个目录。
- 主项目只负责注册目录中暴露的 FastAPI 路由并注入现有数据服务。
- LangChain 负责自然语言条件解析，不直接执行数据库过滤或生成任意绘图代码。
- 日期、灾种、地市、数据类别和强度条件由确定性 Python 代码执行。
- 向量检索用于语义召回，不能绕过显式结构化条件。
- 报告中的事实、统计值、强度值和图片都必须能追溯到命中个例或原始材料。

## 目录

生产代码集中在 `backend/app/services/agent/case_multidim_search/`，包含 schemas、query_parser、structured_retriever、vector_retriever、fusion、intensity、aggregator、chart_tool、exporter、pdf_report、agent 和 router。测试集中在 `tests/case_multidim_search/`。

## 数据流

1. LangChain 结构化输出或确定性降级解析把自然语言转换为 `CaseSearchQuery`。
2. 结构化检索对 `StandardCase` 执行硬过滤。
3. 向量检索从 `ChromaDocumentChunkStore` 召回 chunk，并通过 `source_chunk_ids` 或 `source_pdf` 映射到标准化个例。
4. 融合器对结果去重和排序；显式条件不满足的个例不得进入结果。
5. 强度提取器从个例字段、关联 chunk 和图片附近文本提取带来源的强度指标。
6. 聚合器生成月份、灾种、地市、数据类别和强度统计。
7. 绘图工具只接收经过校验的 `ChartSpec`，输出折线图、柱状图、直方图或散点图。
8. PDF 报告组合检索条件、摘要、统计表、生成图、个例明细和原始证据图。

## API

- `GET /api/case-multidim/health`：返回个例、向量、图片、字体和模型可用状态。
- `POST /api/case-multidim/query`：返回解析条件、检索模式、结果、统计、强度、图片和告警。
- `POST /api/case-multidim/export/csv`：导出当前查询结果 CSV。
- `POST /api/case-multidim/export/xlsx`：导出当前查询结果 XLSX。
- `POST /api/case-multidim/report`：直接返回 `application/pdf`。

## 中心点强度

强度指标包含名称、数值、单位、位置、与中心点的关系、来源 chunk、原文片段和置信度。用户未指定指标时，列出命中个例中所有有证据的中心点强度指标；没有证据时明确说明缺失，禁止推断。

## 报告质量

PDF 使用 Matplotlib `PdfPages`，优先使用项目配置字体，其次使用系统中文字体。生成图和原始证据图必须分区展示。所有图片保持宽高比并带标题、类型、来源 PDF 和页码。报告必须通过文本提取、页面渲染、非空像素、数据一致性和图片相关性检查。

## 验收

五阶段分别验收结构化检索、LangChain 与混合检索、强度提取、统计绘图与表格导出、PDF 与 API。最终运行专项测试、后端回归测试和真实数据报告烟雾测试；无法使用的外部模型必须显式降级并在响应和报告中标注。
