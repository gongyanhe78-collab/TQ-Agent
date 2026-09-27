"""个例多维检索智能体的数据协议。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CaseSearchQuery(BaseModel):
    """由结构化表单提交得到的确定性检索条件。

    多维度正交过滤结构，每个字段代表一个独立的检索维度，
    各维度之间为 AND 关系，用于精确限定检索范围。
    """

    start_date: str = Field(
        default="",
        description="检索起始日期，ISO 8601 格式，如 '2023-07-01'，空字符串表示不限定起始时间",
    )
    end_date: str = Field(
        default="",
        description="检索结束日期，ISO 8601 格式，如 '2023-07-31'，空字符串表示不限定结束时间",
    )
    years: list[int] = Field(
        default_factory=list,
        description="年份过滤列表，如 [2021, 2022, 2023]，用于跨年统计分析，空列表表示不限年份",
    )
    months: list[int] = Field(
        default_factory=list,
        description="月份过滤列表 [1-12]，用于季节性模式检索，如夏季 [6, 7, 8]，空列表表示不限月份",
    )
    disaster_types: list[str] = Field(
        default_factory=list,
        description="灾害类型过滤列表，如 ['暴雨', '冰雹', '雷暴大风']，取值参见 DISASTER_NAMES 常量，空列表表示不限灾种",
    )
    cities: list[str] = Field(
        default_factory=list,
        description="地市过滤列表，如 ['太原', '大同']，取值参见 CITY_NAMES 常量，空列表表示不限地市",
    )
    areas: list[str] = Field(
        default_factory=list,
        description="区域过滤列表，如 ['晋南', '晋北']，用于非行政区域的聚合检索，空列表表示不限区域",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="返回结果最大数量，范围 [1, 500]，用于控制响应大小和检索耗时",
    )


class StructuredCaseSearchRequest(BaseModel):
    """结构化筛选工作台提交的检索与分析请求。"""

    start_date: str = Field(
        default="",
        description="检索起始日期，ISO 8601 格式，空字符串表示不限起始日期",
    )
    end_date: str = Field(
        default="",
        description="检索结束日期，ISO 8601 格式，空字符串表示不限结束日期",
    )
    years: list[int] = Field(
        default_factory=list,
        description="年份过滤列表，空列表表示不限年份",
    )
    months: list[int] = Field(
        default_factory=list,
        description="月份过滤列表，空列表表示不限月份",
    )
    disaster_types: list[str] = Field(
        default_factory=list,
        description="灾种过滤列表，如 强对流、暴雨、雷暴大风",
    )
    cities: list[str] = Field(
        default_factory=list,
        description="地市过滤列表，如 太原、大同、临汾",
    )
    areas: list[str] = Field(
        default_factory=list,
        description="业务区域过滤列表，如 晋北、晋中、晋南、山西北部",
    )
    include_images: bool = Field(
        default=True,
        description="是否返回图片证据",
    )
    include_all_images: bool = Field(
        default=True,
        description="是否为命中个例加载全部可解析图片，默认用于报告完整性保障",
    )
    image_limit_per_case: int = Field(
        default=20,
        ge=0,
        le=100,
        description="未启用全部图片时，单个个例最多返回的图片数量",
    )
    display_limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="页面预览最多展示的个例数量，不影响报告统计的完整命中集合",
    )
    report_include_all_cases: bool = Field(
        default=True,
        description="报告和中间分析是否覆盖全部命中个例，默认开启",
    )
    use_llm_case_analysis: bool = Field(
        default=True,
        description="是否逐个调用大模型分析命中个例；每次调用只包含当前个例的精选 chunk",
    )
    case_chunk_limit: int = Field(
        default=6,
        ge=3,
        le=6,
        description="单个个例最多交给大模型的相关 chunk 数量，固定在 3 至 6 个；不足六个时全部使用",
    )
    case_context_char_limit: int = Field(
        default=6000,
        ge=1000,
        le=12000,
        description="单个个例交给大模型的正文字符上限，用于控制显存和上下文长度",
    )
    case_analysis_concurrency: int = Field(
        default=4,
        ge=1,
        le=8,
        description="逐例生成式大模型的期望并发数；后端实际最多按四路业务流水线控制",
    )
    case_max_output_tokens: int = Field(
        default=1536,
        ge=256,
        le=2048,
        description="单个个例联合输出上限；兼容旧请求，执行阶段统一限制在1200至1536 token",
    )
    progress_id: str = Field(
        default="",
        max_length=80,
        description="前端生成的进度跟踪编号，不参与检索条件匹配",
    )
    report_title: str = Field(
        default="",
        description="报告标题，为空时自动按筛选条件生成",
    )

    def to_query(self) -> CaseSearchQuery:
        """把结构化表单条件转换为检索层使用的标准查询对象。"""
        return CaseSearchQuery(
            start_date=self.start_date,
            end_date=self.end_date,
            years=self.years,
            months=self.months,
            disaster_types=self.disaster_types,
            cities=self.cities,
            areas=self.areas,
            limit=self.display_limit,
        )


class NaturalCaseQueryParseRequest(BaseModel):
    """聊天页面提交的自然语言检索条件解析请求。"""

    message: str = Field(
        min_length=1,
        max_length=2000,
        description="用户本轮自然语言消息",
    )
    conversation_id: str = Field(
        default="",
        max_length=80,
        description="聊天会话编号；首次请求可留空，由后端生成",
    )


class NaturalCaseQueryParseResponse(BaseModel):
    """自然语言解析后的待确认结构化检索提案。"""

    conversation_id: str = Field(description="服务端会话编号")
    proposal_id: str = Field(default="", description="待确认提案编号；不可确认时为空")
    operation: Literal["new", "restrict", "scope", "replace"] = Field(
        description="本轮上下文合并方式",
    )
    operation_label: str = Field(description="面向用户展示的合并方式说明")
    parsed_request: StructuredCaseSearchRequest = Field(description="确认后实际执行的完整结构化请求")
    display_lines: list[str] = Field(default_factory=list, description="分行展示的最终检索条件")
    change_summary: list[str] = Field(default_factory=list, description="相对已确认上下文的修改摘要")
    warnings: list[str] = Field(default_factory=list, description="解析边界和待核对提示")
    can_confirm: bool = Field(default=True, description="当前提案是否允许确认并开始检索")
    parse_status: str = Field(default="rule", description="规则或大模型补充解析状态")


class NaturalCaseSearchRequest(BaseModel):
    """确认自然语言解析提案并执行现有多维检索的请求。"""

    conversation_id: str = Field(min_length=1, max_length=80, description="聊天会话编号")
    proposal_id: str = Field(min_length=1, max_length=80, description="已展示给用户的待确认提案编号")
    progress_id: str = Field(default="", max_length=80, description="检索进度编号")


class NaturalConversationResetRequest(BaseModel):
    """清空自然语言聊天上下文的请求。"""

    conversation_id: str = Field(default="", max_length=80, description="待清空的聊天会话编号")


class IntensityMetric(BaseModel):
    """带原始证据的关键强度指标。

    可溯源数据单元，每个数值都携带来源信息，
    用户可以点击查看原文出处，确保数据的可信度和可验证性。
    """

    metric_name: str = Field(
        description="指标名称，如 '最大风速'、'24小时降水量'、'最高气温'",
    )
    value: float = Field(
        description="指标的数值大小",
    )
    unit: str = Field(
        description="数值单位，如 'mm'(毫米)、'm/s'(米/秒)、'℃'(摄氏度)、'hPa'(百帕)",
    )
    location: str = Field(
        default="",
        description="该强度指标发生的具体地点，如 '晋中市榆社县'",
    )
    relation: str = Field(
        default="",
        description="数值关系描述，如 '最大'、'平均'、'瞬时'、'过程累计'",
    )
    source_chunk_id: str = Field(
        default="",
        description="来源文本块的唯一标识符，可定位到具体 PDF 的具体段落，用于证据追溯",
    )
    source_text: str = Field(
        description="提取该指标的原文句子片段，用于用户验证和人工复核",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="提取置信度，范围 [0.0, 1.0]，规则提取默认为 1.0，模型提取根据置信度调整",
    )


class CaseSearchHit(BaseModel):
    """结构化检索后的单条个例结果。

    "结果 + 证据链" 的复合结构，不仅返回匹配的个例本身，
    还携带完整的匹配证据和提取到的结构化数据，
    这是智能检索与普通全文检索的核心区别。
    """

    case: dict[str, Any] = Field(
        description="个例的完整元数据，包含时间、地点、灾情描述等所有数据库字段，用 dict 便于灵活扩展",
    )
    score: float = Field(
        description="结构化条件匹配得分，用于结果排序",
    )
    matched_fields: list[str] = Field(
        default_factory=list,
        description="命中的字段名称列表，如 ['cities', 'disaster_types']，用于向用户解释匹配原因",
    )
    analysis: str = Field(
        default="",
        description="面向报告正文的单个个例分析，不暴露底层存储编号",
    )
    intensity_metrics: list[IntensityMetric] = Field(
        default_factory=list,
        description="从该个例中提取到的结构化强度指标列表，支持后续的统计分析和图表绘制",
    )
    evidence_images: list[dict[str, Any]] = Field(
        default_factory=list,
        description="图片证据列表，包含雷达图、卫星云图、降水图等，每张图带 URL、标题、页码、说明等信息",
    )


class ChartSpec(BaseModel):
    """绘图工具接受的白名单图表配置。

    LLM 与绘图工具之间的协议，采用白名单设计限制允许的图表类型，
    字段设计简单扁平，确保 LLM 能够稳定生成正确的结构。
    """

    chart_key: str = Field(
        default="",
        description="图表所属分析章节键值，如 temporal、disaster、spatial、intensity，便于前端和 PDF 精确挂接",
    )
    chart_type: Literal["line", "bar", "horizontal_bar", "progress_bar", "lollipop", "histogram", "boxplot", "pie", "scatter", "heatmap"] = Field(
        description="图表类型白名单，仅支持：line(折线图)、bar(柱状图)、horizontal_bar(横向条形图)、lollipop(点棒排名图)、histogram(直方图)、boxplot(箱线图)、pie(饼图)、scatter(散点图)、heatmap(热力图)",
    )
    title: str = Field(
        description="图表标题，简洁描述图表内容，如 '2023年各月暴雨过程次数统计'",
    )
    x_label: str = Field(
        description="X 轴标签，说明横轴代表的含义，如 '月份'、'降水量区间'",
    )
    y_label: str = Field(
        description="Y 轴标签，说明纵轴代表的含义，如 '过程次数'、'个例数'",
    )
    labels: list[str] = Field(
        default_factory=list,
        description="分类标签数组，用于柱状图和直方图，如 ['1月', '2月', '3月']",
    )
    values: list[float] = Field(
        default_factory=list,
        description="Y 值数组，柱状图/直方图的数值序列，长度应与 labels 一一对应",
    )
    row_labels: list[str] = Field(
        default_factory=list,
        description="热力图的行标签，通常对应灾种共现矩阵中的灾种名称",
    )
    col_labels: list[str] = Field(
        default_factory=list,
        description="热力图的列标签，通常与 row_labels 保持一致",
    )
    matrix: list[list[float]] = Field(
        default_factory=list,
        description="热力图对应的二维数值矩阵，每个元素都应经过统计标准化后输出",
    )
    x_values: list[float] = Field(
        default_factory=list,
        description="X 坐标值数组，用于散点图和折线图的横坐标定位",
    )
    y_values: list[float] = Field(
        default_factory=list,
        description="Y 坐标值数组，用于散点图和折线图的纵坐标定位，长度应与 x_values 相同",
    )
    unit: str = Field(
        default="",
        description="数值的单位，如 'mm'、'次'、'%'，展示在图表的适当位置",
    )
    sample_size: int = Field(
        default=0,
        description="统计的样本总量，用于说明数据的统计显著性，0 表示未统计",
    )
    missing_count: int = Field(
        default=0,
        description="缺失值数量，用于说明数据质量，让用户了解统计结果的完整性",
    )
    interpretation: str = Field(
        default="",
        description="图表解读文字，说明该图反映的主要分布特征和业务含义",
    )


class CaseSearchResponse(BaseModel):
    """Postman 查询接口返回的完整数据和分析。

    API 层的最终输出协议，是一个"数据 + 解释 + 可视化 + 元信息"的完整响应包，
    包含了前端进行丰富交互所需的全部信息。
    """

    question: str = Field(
        description="结构化检索条件的可读描述，回显给用户确认系统正确接收了查询内容",
    )
    parsed_query: CaseSearchQuery = Field(
        description="表单提交得到的结构化检索条件，便于确认或修正",
    )
    retrieval_mode: str = Field(
        description="检索模式标识，结构化表单检索固定为 'structured_full'",
    )
    result_count: int = Field(
        description="满足条件的个例总命中数，可能大于实际返回的数量（受 limit 限制或分页）",
    )
    displayed_result_count: int = Field(
        default=0,
        description="本次响应实际返回的结果数量，分页场景下小于等于 result_count",
    )
    answer: str = Field(
        description="Agent 生成的分析总结，对结构化检索结果进行概括",
    )
    analysis: dict[str, Any] = Field(
        default_factory=dict,
        description="额外的结构化分析结论，如时间分布特征、灾害类型占比、极端值统计等，键名即为分析项名称",
    )
    results: list[CaseSearchHit] = Field(
        description="个例检索结果列表，每个结果携带完整的元数据和证据链，按相关性从高到低排序",
    )
    aggregations: dict[str, Any] = Field(
        description="聚合统计数据，按各维度（年份、月份、灾种、地市等）的分布统计结果，用于绘制分布图和统计表",
    )
    charts: list[ChartSpec] = Field(
        default_factory=list,
        description="图表配置列表，前端可直接渲染的标准化图表规范，为空表示无图表",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="警告信息列表，如 LLM 降级提示、数据缺失说明、检索边界警告等，让用户了解系统行为",
    )
    audit: dict[str, Any] = Field(
        default_factory=dict,
        description="审计和调试信息，包含各阶段耗时、模型调用次数、检索参数等，用于排障、优化和性能监控",
    )


class CasePdfExportRequest(BaseModel):
    """按已生成分析结果导出 PDF 的请求。"""

    answer_id: str = Field(
        min_length=1,
        description="分析接口返回的回答编号，用于定位已确认的分析内容",
    )
    filename: str = Field(
        default="case-search-report.pdf",
        description="下载 PDF 时使用的文件名，留空时使用默认报告名称",
    )
    report_title: str = Field(
        default="",
        description="导出 PDF 时覆盖使用的报告标题，留空时沿用分析阶段标题",
    )

    progress_id: str = Field(
        default="",
        max_length=80,
        description="PDF 导出进度编号，用于前端轮询导出状态",
    )


class CaseAnalysisSection(BaseModel):
    """面向前端或 Postman 展示的报告章节。"""

    heading: str = Field(
        description="章节标题，用于展示自然语言分析结构",
    )
    content: str = Field(
        description="章节正文，要求是分析结论而不是字段罗列",
    )


class CaseAnalysisVisual(BaseModel):
    """分析结果中的图片或图表资源。"""

    visual_id: str = Field(
        description="图片或图表的唯一编号",
    )
    visual_type: Literal["database_image", "generated_chart"] = Field(
        description="资源类型，database_image 表示数据库原图，generated_chart 表示智能体生成图表",
    )
    title: str = Field(
        description="资源标题或图注",
    )
    chart_key: str = Field(
        default="",
        description="图表所属章节键值，如 temporal、disaster、spatial、intensity；图片资源留空",
    )
    url: str = Field(
        description="可通过后端访问的资源 URL",
    )
    path: str = Field(
        default="",
        description="资源在本机的实际路径，便于 Postman 调试和人工核对",
    )
    source: str = Field(
        default="",
        description="资源来源，如 PDF 文件名、个例编号或图表生成说明",
    )


class CaseAnalysisResponse(BaseModel):
    """先返回给用户审阅的完整智能体分析结果。"""

    answer_id: str = Field(
        description="本次分析回答编号，后续 PDF 导出接口使用该编号",
    )
    title: str = Field(
        description="分析标题，PDF 导出时默认沿用该标题",
    )
    question: str = Field(
        description="用户原始自然语言问题",
    )
    answer: str = Field(
        description="可直接阅读的总体回答",
    )
    sections: list[CaseAnalysisSection] = Field(
        default_factory=list,
        description="分章节分析正文",
    )
    images: list[CaseAnalysisVisual] = Field(
        default_factory=list,
        description="数据库中命中的原始图片证据",
    )
    charts: list[CaseAnalysisVisual] = Field(
        default_factory=list,
        description="智能体根据统计结果生成的图表",
    )
    evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="可追溯证据摘要，说明命中的个例、来源和命中原因",
    )
    search_response: CaseSearchResponse = Field(
        description="完整检索响应，保留给前端或测试阶段做深度核对",
    )
    can_export_pdf: bool = Field(
        default=True,
        description="是否可以继续调用 PDF 导出接口",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="分析阶段产生的警告信息",
    )
    audit: dict[str, Any] = Field(
        default_factory=dict,
        description="分析阶段审计信息，包含 answer_id、资源数量和检索链路统计",
    )








