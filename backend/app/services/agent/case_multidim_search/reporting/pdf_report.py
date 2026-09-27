"""生成参考技术报告版式的中文个例分析 PDF。"""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path


from backend.app.services.agent.case_multidim_search.reporting.chart_tool import ChartTool
from backend.app.services.agent.case_multidim_search.reporting.report_document import ReferenceReportDocument
from backend.app.services.agent.case_multidim_search.schemas import CaseSearchResponse


class PdfReportBuilder:
    """把检索分析、统计图和原始证据编排为连续技术报告。"""

    def __init__(self, chart_tool: ChartTool | None = None):
        """初始化报告使用的统计绘图工具。"""
        self.chart_tool = chart_tool or ChartTool()

    def build(
        self,
        response: CaseSearchResponse,
        output_path: Path,
        report_title: str = "",
        progress_callback=None,
        chart_paths: list[Path] | None = None,
    ) -> Path:
        """生成白底 A4、文字分析优先且图文对应的正式 PDF。"""
        import matplotlib

        logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
        matplotlib.use("Agg")
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        title = report_title or "山西省气象灾害个例多维分析报告"
        self._notify_progress(progress_callback, 12, "正在渲染报告图表")
        reusable_paths = [Path(path) for path in (chart_paths or []) if Path(path).is_file()]
        if len(reusable_paths) == len(response.charts):
            chart_paths = reusable_paths
        else:
            # 旧缓存没有网页图表资产时才降级重绘，保证历史回答仍可导出。
            chart_paths = [
                self.chart_tool.render(spec, output_path.parent / f"{output_path.stem}-chart-{index}.png")
                for index, spec in enumerate(response.charts, start=1)
            ]
        with PdfPages(output_path) as pdf:
            pdf.infodict()["Title"] = title
            pdf.infodict()["Subject"] = "气象灾害个例多维检索分析"
            document = ReferenceReportDocument(pdf, plt, self.chart_tool._font_path())
            document.add_report_title(title, f"生成日期：{date.today().strftime('%Y年%m月%d日')}", subtitle=self._report_subtitle(response))
            self._notify_progress(progress_callback, 30, "正在写入综合概况和强度汇总")
            self._write_overview(document, response)
            self._notify_progress(progress_callback, 50, "正在写入分维度分析和图表")
            self._write_main_analysis(document, response, chart_paths)
            self._notify_progress(progress_callback, 70, "正在写入代表个例分析和原图")
            self._write_cases(document, response)
            self._notify_progress(progress_callback, 88, "正在写入综合结论与建议")
            self._write_conclusion(document, response)
            self._notify_progress(progress_callback, 96, "正在保存 PDF 文件")
            document.finish()
        return output_path


    def _report_subtitle(self, response: CaseSearchResponse) -> str:
        """把结构化检索条件分行放入副标题，避免筛选词挤在同一行。"""
        query = getattr(response, "parsed_query", None)
        if query is not None:
            lines = []
            start_date = str(getattr(query, "start_date", "") or "").strip()
            end_date = str(getattr(query, "end_date", "") or "").strip()
            years = list(getattr(query, "years", []) or [])
            months = list(getattr(query, "months", []) or [])
            disasters = list(getattr(query, "disaster_types", []) or [])
            cities = list(getattr(query, "cities", []) or [])
            areas = list(getattr(query, "areas", []) or [])
            if start_date or end_date:
                lines.append(f"日期：{start_date or '不限'} 至 {end_date or '不限'}")
            if years:
                lines.append("年份：" + "、".join(f"{year}年" for year in years))
            if months:
                lines.append("月份：" + "、".join(f"{month}月" for month in months))
            if disasters:
                lines.append("灾种：" + "、".join(str(item) for item in disasters))
            if cities or areas:
                lines.append("地市：" + "、".join(str(item) for item in [*cities, *areas]))
            if lines:
                return "检索条件：\n" + "\n".join(lines)
        question = str(getattr(response, "question", "") or "").strip()
        return f"检索条件：{question}" if question else ""

    def _notify_progress(self, progress_callback, percent: int, message: str) -> None:
        """安全通知 PDF 导出进度，避免进度回调异常影响文件生成。"""
        if not callable(progress_callback):
            return
        try:
            progress_callback(percent, message)
        except Exception:
            return

    def _write_overview(self, document: ReferenceReportDocument, response: CaseSearchResponse) -> None:
        """写入首页综合概况、重点发现和对应的代表图像。"""
        analysis = response.analysis or {}
        document.add_section("综合概况")
        document.add_paragraph(analysis.get("executive_summary") or response.answer)
        self._write_overview_intensity_table(document, analysis)
        findings = analysis.get("key_findings") or []
        if findings:
            document.add_paragraph(self._findings_paragraph(findings))

    def _write_overview_intensity_table(self, document: ReferenceReportDocument, analysis: dict) -> None:
        """在 PDF 概况后写入分组强度汇总，字段与网页表格同源。"""
        tables = analysis.get("overview_intensity_tables") or []
        if tables:
            document.add_subsection("重点个例强度汇总")
            wrote_table = False
            for table in tables:
                rows = table.get("rows") or []
                columns = (table.get("columns") or [])[:3]
                if not rows or not columns:
                    continue
                document.add_paragraph(str(table.get("title") or "强度汇总"), first_line_indent=False)
                headers = ["个例", "主导灾种", "并发灾害"] + [str(column.get("label") or column.get("key")) for column in columns]
                table_rows = []
                for row in rows[:12]:
                    metrics = row.get("metrics") or {}
                    metric_values = [str(metrics.get(column.get("key"), "—")) for column in columns]
                    table_rows.append([
                        str(row.get("case") or row.get("title") or row.get("case_id") or "—"),
                        str(row.get("main_disasters") or "—"),
                        str(row.get("concurrent_disasters") or "—"),
                        *metric_values,
                    ])
                document.add_table(headers, table_rows, max_rows=12)
                wrote_table = True
            if wrote_table:
                document.add_paragraph("注：“—”表示原文未提供有效记录；表中强度数值均来自对应天气过程的正式业务材料，疑似异常值不作为可信极值展示。", first_line_indent=False)
            return

        rows = analysis.get("overview_intensity_table") or []
        columns = (analysis.get("overview_intensity_columns") or [])[:3]
        if not rows or not columns:
            return
        document.add_subsection("重点个例强度汇总")
        headers = ["个例", "主导灾种", "并发灾害"] + [str(column.get("label") or column.get("key")) for column in columns]
        table_rows = []
        for row in rows[:12]:
            metrics = row.get("metrics") or {}
            metric_values = [str(metrics.get(column.get("key"), "—")) for column in columns]
            table_rows.append([
                str(row.get("case") or row.get("title") or row.get("case_id") or "—"),
                str(row.get("main_disasters") or "—"),
                str(row.get("concurrent_disasters") or "—"),
                *metric_values,
            ])
        document.add_table(headers, table_rows, max_rows=12)
        document.add_paragraph("注：“—”表示原文未提供有效记录；表中强度数值均来自对应天气过程的正式业务材料。", first_line_indent=False)

    def _write_main_analysis(
        self,
        document: ReferenceReportDocument,
        response: CaseSearchResponse,
        chart_paths: list[Path],
    ) -> None:
        """按时间、灾种、空间、强度和证据章节写入分析及对应图表。"""
        analysis = response.analysis or {}
        sections = analysis.get("sections") or {}
        chart_map = self._chart_paths_by_section(response, chart_paths)
        document.add_section("一、主要分析")
        section_items = [
            ("temporal", "1. 时间分布特征"),
            ("disaster", "2. 灾种结构特征"),
            ("spatial", "3. 空间影响特征"),
            ("intensity", "4. 关键强度指标特征"),
        ]
        for key, title in section_items:
            body = sections.get(key)
            if not body:
                continue
            document.add_subsection(title)
            document.add_paragraph(body)
            for path, caption, insight in chart_map.get(key, []):
                document.add_figure(path, caption, max_height=0.34, numbered=False)
                if insight:
                    document.add_paragraph(insight, first_line_indent=True)

    def _write_cases(self, document: ReferenceReportDocument, response: CaseSearchResponse) -> None:
        """连续编排代表个例的过程特征、强度、研判和数据库原图。"""
        document.add_section("二、代表个例分析")
        if not response.results:
            document.add_paragraph("当前查询未命中可用于展开分析的标准化个例。")
            return
        for index, hit in enumerate(response.results, start=1):
            case = hit.case
            title = case.get("title") or case.get("case_id") or f"个例{index}"
            document.add_subsection(f"{index}. {title}")
            document.add_paragraph(
                self._clean_analysis_text(hit.analysis) or self._case_analysis(case, hit.intensity_metrics, image_count=len(hit.evidence_images))
            )
            valid_image_count = 0
            for image_data in hit.evidence_images:
                image_path = Path(str(image_data.get("image_path", "")))
                caption = self._image_caption(image_data)
                if document.add_figure(
                    image_path,
                    caption,
                    figure_number=image_data.get("display_number"),
                ):
                    valid_image_count += 1
            if hit.evidence_images and valid_image_count == 0:
                document.add_paragraph("该个例已关联图片记录，但当前文件无法正常解码，相关图像证据需进一步核查。")
    def _clean_analysis_text(self, value: str) -> str:
        """清理旧缓存中直接暴露的 JSON 协议，只保留可读的自然语言分析。"""
        text = str(value or "").strip()
        if not text.startswith("{") or "analysis" not in text:
            return text
        try:
            payload = json.loads(text)
            return str(payload.get("analysis") or text).strip()
        except json.JSONDecodeError:
            match = re.search(r'"analysis"\s*:\s*"([\s\S]*?)"\s*,\s*"cited_image_ids"', text)
            if not match:
                return text
            return match.group(1).replace(r"\n", "\n").replace(r'\"', '"').strip()

    def _write_conclusion(self, document: ReferenceReportDocument, response: CaseSearchResponse) -> None:
        """写入综合结论和业务建议。"""
        analysis = response.analysis or {}
        document.add_section("三、综合结论与建议")
        document.add_subsection("1. 综合结论")
        document.add_paragraph(analysis.get("conclusion") or response.answer)
        recommendations = analysis.get("recommendations") or []
        if recommendations:
            document.add_subsection("2. 业务建议")
            document.add_numbered_items(recommendations)

    def _chart_paths_by_section(
        self,
        response: CaseSearchResponse,
        chart_paths: list[Path],
    ) -> dict[str, list[tuple[Path, str, str]]]:
        """把统计图按含义映射到对应分析章节，每章允许保留多张。"""
        mapped: dict[str, list[tuple[Path, str, str]]] = {}
        for spec, path in zip(response.charts, chart_paths):
            key = str(getattr(spec, "chart_key", "") or "").strip()
            section_key = self._chart_section(key, spec.x_label, spec.title)
            if not section_key:
                continue
            mapped.setdefault(section_key, []).append((path, self._chart_caption(spec), str(getattr(spec, "interpretation", "") or "")))
        return mapped

    def _chart_caption(self, spec) -> str:
        """PDF 图注只保留本报告图名，业务启示另起正文段落。"""
        return str(spec.title)

    def _chart_section(self, chart_key: str, x_label: str, title: str) -> str:
        """根据图表键名或标签判断其对应的分析章节。"""
        key = str(chart_key or "").strip()
        if key == "temporal":
            return "temporal"
        if key.startswith("disaster"):
            return "disaster"
        if key == "spatial":
            return "spatial"
        if key == "intensity" or key.startswith("intensity_"):
            return "intensity"
        if x_label == "月份":
            return "temporal"
        if x_label == "灾种":
            return "disaster"
        if x_label == "地市":
            return "spatial"
        if "分布" in str(title):
            return "intensity"
        return ""

    def _findings_paragraph(self, findings: list[str]) -> str:
        """把重点发现合并为适合连续阅读的编号分析段落。"""
        parts = [f"（{index}）{str(item).rstrip('。')}" for index, item in enumerate(findings, start=1)]
        return "主要发现包括：" + "；".join(parts) + "。"

    def _case_analysis(self, case: dict, metrics: list, image_count: int = 0) -> str:
        """根据结构化事实归纳个例过程、关键强度指标和业务研判。"""
        overview = self._case_overview(case, image_count=image_count)
        if metrics:
            metric_parts = []
            for metric in metrics:
                location = f"（{metric.location}）" if metric.location else ""
                relation = metric.relation or "为"
                metric_parts.append(
                    f"{metric.metric_name}{relation}{metric.value:g}{metric.unit}{location}"
                )
            intensity = "；".join(metric_parts) + "。"
        else:
            intensity = "现有业务材料未明确给出可核验的关键强度指标数值。"
        return f"过程特征：{overview} 关键强度指标：{intensity} 综合研判：{self._case_focus(case)}"

    def _case_overview(self, case: dict, image_count: int) -> str:
        """根据结构化日期、灾种、区域和证据数量生成干净的个例概况。"""
        date_range = str(case.get("date_range") or "时间未明确")
        disasters = "、".join(case.get("disaster_types") or []) or "未标注灾种"
        areas = case.get("city_tags") or case.get("affected_areas") or []
        area_text = "、".join(areas) if isinstance(areas, list) else str(areas)
        area_text = area_text or "影响区域未明确"
        evidence_text = (
            f"本次报告关联 {image_count} 张原始证据图，可结合图像进一步核查过程落区和强度变化。"
            if image_count > 0
            else "本次报告未关联到可展示的原始证据图，相关判断需结合后续补充材料复核。"
        )
        return f"{date_range}，{area_text}发生以{disasters}为主的天气过程。{evidence_text}"

    def _case_focus(self, case: dict) -> str:
        """根据个例灾种归纳风险重点而不是复制原文目录。"""
        disasters = set(case.get("disaster_types") or [])
        focuses = []
        if disasters.intersection({"暴雨", "大暴雨", "强降水", "短时强降水"}):
            focuses.append("核查累计降水和短时雨强落区，关注城市内涝、中小河流洪水及地质灾害风险")
        if disasters.intersection({"强对流", "雷暴", "雷暴大风", "冰雹", "大风"}):
            focuses.append("跟踪雷达回波和地面观测演变，关注雷暴大风、冰雹及短时强降水的突发影响")
        if disasters.intersection({"高温", "低温", "寒潮", "暴雪", "雨雪"}):
            focuses.append("关注温度变化、道路结冰和能源保供等持续性影响")
        return "；".join(focuses) + "。" if focuses else "结合原始证据继续核查过程落区、强度和影响。"

    def _image_caption(self, image_data: dict) -> str:
        """PDF 图注只展示原始图题，连续图号由报告排版层统一生成。"""
        caption = image_data.get("display_caption") or image_data.get("caption") or image_data.get("image_id") or "原始证据图"
        return str(caption)



