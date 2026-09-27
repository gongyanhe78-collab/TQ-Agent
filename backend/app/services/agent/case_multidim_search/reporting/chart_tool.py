"""基于白名单图表规格生成报告统计图片。"""
from __future__ import annotations

from pathlib import Path

from backend.app.services.agent.case_multidim_search.analysis.disaster_profile import (
    dominant_disasters_from_query_or_counts,
    profiles_for_disasters,
)
from backend.app.services.agent.case_multidim_search.schemas import ChartSpec


class ChartTool:
    """只接收结构化数据，不执行模型生成的绘图代码。"""

    def select_specs(self, aggregations: dict, query=None) -> list[ChartSpec]:
        """根据聚合结果选择更适合业务表达的图表类型。"""
        specs: list[ChartSpec] = []

        month_counts = aggregations.get("month_counts", {})
        if len(month_counts) >= 2:
            # 时间序列仍然适合折线图，但要保持单色系，避免视觉噪音太大。
            specs.append(
                ChartSpec(
                    chart_key="temporal",
                    chart_type="line",
                    title="个例数量月度变化",
                    x_label="月份",
                    y_label="个例数量",
                    labels=[f"{month}月" for month in month_counts],
                    values=[float(value) for value in month_counts.values()],
                    sample_size=sum(month_counts.values()),
                )
            )

        case_count = int(aggregations.get("case_count") or 0)
        disaster_counts = aggregations.get("disaster_counts", {})
        if disaster_counts:
            # 频次图按次数降序排列，采用水平柱状图避免灾种标签挤压或竖排重叠。
            ordered_disasters = sorted(
                ((str(name), float(value)) for name, value in disaster_counts.items()),
                key=lambda item: (-item[1], item[0]),
            )
            labels = [name for name, _ in ordered_disasters]
            values = [value for _, value in ordered_disasters]
            specs.append(
                ChartSpec(
                    chart_key="disaster_frequency",
                    chart_type="horizontal_bar",
                    title="灾种频次分布",
                    x_label="出现频次",
                    y_label="",
                    labels=labels,
                    values=values,
                    sample_size=case_count or int(sum(values)),
                )
            )
            cooccurrence = aggregations.get("disaster_cooccurrence", {}) or {}
            heat_labels, matrix = self._sorted_cooccurrence_data(disaster_counts, cooccurrence)
            if len(heat_labels) >= 2 and matrix:
                specs.append(
                    ChartSpec(
                        chart_key="disaster_cooccurrence",
                        chart_type="heatmap",
                        title="灾种共现热力图",
                        x_label="灾种",
                        y_label="灾种",
                        row_labels=heat_labels,
                        col_labels=heat_labels,
                        matrix=matrix,
                        sample_size=case_count or int(sum(values)),
                    )
                )
        city_counts = aggregations.get("city_counts", {})
        if city_counts:
            ordered_cities = sorted(
                ((str(name), float(value)) for name, value in city_counts.items()),
                key=lambda item: (-item[1], item[0]),
            )[:8]
            # 地市可以在同一个例中同时出现，因此展示命中个例数排名，不再计算互斥占比。
            specs.append(
                ChartSpec(
                    chart_key="spatial",
                    chart_type="horizontal_bar",
                    title="主要影响区域命中个例数",
                    x_label="命中个例数",
                    y_label="影响区域",
                    labels=[name for name, _value in ordered_cities],
                    values=[value for _name, value in ordered_cities],
                    sample_size=case_count or int(max((value for _name, value in ordered_cities), default=0)),
                )
            )

        focus_disasters = dominant_disasters_from_query_or_counts(query, aggregations)
        disaster_specific_specs = self._disaster_specific_specs(aggregations, focus_disasters)
        if disaster_specific_specs:
            specs.extend(disaster_specific_specs)
        elif not profiles_for_disasters(focus_disasters):
            intensity_values = aggregations.get("intensity_values", {}) or {}
            if intensity_values:
                metric_name, values = max(intensity_values.items(), key=lambda item: len(item[1]))
                clean_values = [float(value) for value in values]
                if len(clean_values) >= 4:
                    chart_type = "boxplot"
                    title = f"{metric_name}分布箱线图"
                else:
                    chart_type = "histogram"
                    title = f"{metric_name}分布"
                specs.append(
                    ChartSpec(
                        chart_key="intensity",
                        chart_type=chart_type,
                        title=title,
                        x_label=metric_name,
                        y_label="数值",
                        values=clean_values,
                        sample_size=len(clean_values),
                    )
                )

        selected_specs = specs[:8]
        for spec in selected_specs:
            spec.interpretation = self.interpret(spec)
        return selected_specs

    def _sorted_cooccurrence_data(self, disaster_counts: dict, cooccurrence: dict) -> tuple[list[str], list[list[float]]]:
        """按灾种频次重排共现矩阵，让高频灾种集中在图表左上方。"""
        source_labels = [str(item) for item in (cooccurrence.get("labels") or []) if str(item)]
        source_matrix = cooccurrence.get("matrix") or []
        if not source_labels or not source_matrix:
            return [], []
        index_by_label = {label: index for index, label in enumerate(source_labels)}
        ordered_labels = sorted(
            source_labels,
            key=lambda label: (-float(disaster_counts.get(label, 0)), label),
        )
        matrix: list[list[float]] = []
        for row_label in ordered_labels:
            row_index = index_by_label[row_label]
            source_row = source_matrix[row_index] if row_index < len(source_matrix) else []
            matrix.append(
                [
                    float(source_row[index_by_label[col_label]])
                    if index_by_label[col_label] < len(source_row)
                    else 0.0
                    for col_label in ordered_labels
                ]
            )
        return ordered_labels, matrix
    def _disaster_specific_specs(self, aggregations: dict, disaster_names: list[str]) -> list[ChartSpec]:
        """按页面勾选灾种生成特色图；每类灾种最多一张，数据少于 3 个例时不画。"""
        case_values = aggregations.get("intensity_case_values", {}) or {}
        specs: list[ChartSpec] = []
        used_keys: set[str] = set()
        for profile in profiles_for_disasters(disaster_names):
            profile_specs = self._profile_charts(profile, case_values)
            for spec in profile_specs:
                if spec.chart_key in used_keys:
                    continue
                specs.append(spec)
                used_keys.add(spec.chart_key)
                break
            if len(specs) >= 4:
                break
        return specs

    def _profile_charts(self, profile, case_values: dict) -> list[ChartSpec]:
        """为单个灾种视角按指标优先级生成候选图表，至少 3 个例才返回。"""
        charts: list[ChartSpec] = []
        # 灾种特色图表统一采用单指标水平柱状图，不再生成复合散点图。
        for metric in profile.metrics:
            rows = self._top_case_metric_rows(case_values.get(metric.metric_name, []), metric.metric_name)
            distinct_values = {float(row.get("value") or 0) for row in rows}
            # 特色图只在同一指标至少覆盖 3 个例且存在差异时展示；少量数据交给文字解释即可。
            if len(rows) < 3 or len(distinct_values) < 2:
                continue
            charts.append(
                ChartSpec(
                    chart_key=metric.chart_key,
                    # 灾种特色图表统一使用标准水平柱状图，便于横向对比并保持 PDF 中清晰可读。
                    chart_type="horizontal_bar",
                    title=metric.chart_title,
                    x_label=f"{metric.metric_name}（{metric.unit}）",
                    y_label="个例",
                    labels=[self._case_axis_label(row.get("title", "")) for row in rows],
                    values=[float(row.get("value") or 0) for row in rows],
                    unit=metric.unit,
                    sample_size=len(rows),
                )
            )
        return charts

    def _top_case_metric_rows(self, rows: list[dict], metric_name: str, limit: int = 8) -> list[dict]:
        """同一指标同一个例可能有多条证据，按业务含义保留最具代表性的极值。"""
        by_case: dict[str, dict] = {}
        prefer_min = "最低" in metric_name or "能见度" in metric_name
        for row in rows:
            case_id = str(row.get("case_id") or row.get("title") or "")
            if not case_id:
                continue
            value = float(row.get("value") or 0)
            current = by_case.get(case_id)
            if current is None:
                by_case[case_id] = dict(row)
                continue
            current_value = float(current.get("value") or 0)
            if (prefer_min and value < current_value) or (not prefer_min and value > current_value):
                by_case[case_id] = dict(row)
        return sorted(
            by_case.values(),
            key=lambda item: float(item.get("value") or 0),
            reverse=not prefer_min,
        )[:limit]

    def _paired_metric_points(self, case_values: dict, x_metric: str, y_metric: str) -> list[tuple[str, float, float]]:
        """为强对流复合灾种构造雨强和风速的同个例散点。"""
        x_rows = {row.get("case_id"): row for row in self._top_case_metric_rows(case_values.get(x_metric, []), x_metric, 99)}
        y_rows = {row.get("case_id"): row for row in self._top_case_metric_rows(case_values.get(y_metric, []), y_metric, 99)}
        points = []
        for case_id, x_row in x_rows.items():
            y_row = y_rows.get(case_id)
            if not y_row:
                continue
            points.append((self._short_label(str(x_row.get("title") or case_id)), float(x_row.get("value") or 0), float(y_row.get("value") or 0)))
        return points[:8]

    def _short_label(self, value: str, limit: int = 18) -> str:
        """压缩过长标题，避免横向条形图标签挤占画布。"""
        text = str(value or "").strip()
        return text if len(text) <= limit else text[:limit].rstrip() + "…"

    def _case_axis_label(self, value: str) -> str:
        """灾种特色图保留完整事件名称，只做空白归一化，不再用省略号截断。"""
        return " ".join(str(value or "").strip().split())

    def interpret(self, spec: ChartSpec) -> str:
        """根据图表数据生成适合正文引用的中文解读。"""
        if spec.chart_key == "intensity_convection_scatter" and spec.x_values and spec.y_values:
            max_rain = max(spec.x_values)
            max_wind = max(spec.y_values)
            return f"该图把强对流个例的小时雨强与极大风速放在同一坐标系中，最大雨强约{max_rain:g}mm/h，最大极大风速约{max_wind:g}m/s，可辅助识别雨强和风速共同突出的代表过程。"

        if spec.chart_key.startswith("intensity_") and spec.labels and spec.values:
            pairs = list(zip(spec.labels, spec.values))
            prefer_min = "最低" in spec.title or "能见度" in spec.title
            focus_label, focus_value = min(pairs, key=lambda item: item[1]) if prefer_min else max(pairs, key=lambda item: item[1])
            unit = spec.unit or ""
            verb = "最低" if prefer_min else "最高"
            return f"该图按个例对比{spec.x_label}，{focus_label}的{verb}值为{focus_value:g}{unit}，适合从当前命中样本中识别需要优先复核的代表过程。"
        if spec.chart_type == "line" and spec.labels and spec.values:
            pairs = list(zip(spec.labels, spec.values))
            top_label, _top_value = max(pairs, key=lambda item: item[1])
            # 图面已经展示具体频次，这里只保留业务启示，避免报告文字重复报数。
            return f"业务启示：{top_label}是本次样本的时间集中窗口，复盘时应重点对照该时段的天气背景和服务响应。"

        if spec.chart_type in {"bar", "horizontal_bar", "progress_bar", "lollipop"} and spec.labels and spec.values:
            pairs = list(zip(spec.labels, spec.values))
            top_label, _top_value = max(pairs, key=lambda item: item[1])
            # 分布图的数值由图面承担，文字只解释它对业务筛查的意义。
            if spec.chart_key == "spatial":
                return f"业务启示：{top_label}可作为空间复核的优先区域，后续应结合地形、站点密度和灾情记录判断集中原因。"
            if spec.chart_key == "disaster_frequency":
                return f"业务启示：{top_label}是当前检索条件下最需要优先组织证据链的灾种，其他伴随灾种应放在复合风险中解释。"
            if spec.chart_key == "evidence":
                return f"业务启示：{top_label}材料占优，说明证据链条的复核重点应放在该类资料的一致性和可追溯性上。"
            return f"业务启示：{top_label}是该维度下最突出的筛查入口，可用于快速定位代表个例。"

        if spec.chart_type == "heatmap" and spec.row_labels and spec.matrix:
            best_pair = self._strongest_pair(spec.row_labels, spec.matrix)
            if best_pair:
                left, right, _value = best_pair
                return f"业务启示：{left}与{right}更容易在同一过程中叠加出现，报告研判应解释触发条件和影响落区是否同步。"
            return "业务启示：该热力图用于识别复合灾种组合，颜色较深的位置应作为个例复核重点。"

        if spec.chart_type == "pie" and spec.labels and spec.values:
            pairs = list(zip(spec.labels, spec.values))
            top_label, top_value = max(pairs, key=lambda item: item[1])
            total = sum(spec.values)
            share = top_value / total * 100 if total else 0
            return f"{top_label}在该互斥分类的样本构成中占比最高，约为{share:.1f}%，应结合分类口径解释其业务含义。"

        if spec.chart_type == "histogram" and spec.values:
            minimum = min(spec.values)
            maximum = max(spec.values)
            average = sum(spec.values) / len(spec.values)
            return f"该图展示{spec.x_label}的样本分布，数值范围为{minimum:g}至{maximum:g}，平均约{average:.1f}。"

        if spec.chart_type == "boxplot" and spec.values:
            minimum = min(spec.values)
            maximum = max(spec.values)
            median = self._median(spec.values)
            return f"该图展示{spec.x_label}的箱线分布，中位数约{median:.1f}，范围为{minimum:g}至{maximum:g}。"

        if spec.chart_type == "scatter" and spec.x_values and spec.y_values:
            return "该图用于观察两个指标之间的对应关系。"

        return "该图用于辅助说明当前检索样本的统计结构。"

    def render(self, spec: ChartSpec, output_path: Path) -> Path:
        """把结构化图表规格渲染为白底高清统计图片。"""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.font_manager import FontProperties

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        font_path = self._font_path()
        times_font_path = self._times_font_path()
        # 中文使用宋体加粗，数字和英文优先使用 Times New Roman 加粗。
        font = FontProperties(fname=str(font_path)) if font_path else FontProperties(family="SimSun")
        font.set_weight("bold")
        bold_font = font.copy()
        bold_font.set_weight("bold")
        number_font = FontProperties(fname=str(times_font_path)) if times_font_path else FontProperties(family="Times New Roman")
        number_font.set_weight("bold")
        progress_font = font.copy()
        progress_font.set_weight("normal")
        progress_number_font = number_font.copy()
        progress_number_font.set_weight("normal")
        plt.rcParams.update(
            {
                "font.weight": "bold",
                "axes.titleweight": "bold",
                "axes.labelweight": "bold",
                "axes.titlesize": 28,
                "axes.labelsize": 24,
                "xtick.labelsize": 22,
                "ytick.labelsize": 22,
                "axes.unicode_minus": False,
            }
        )

        figure, axis = self._create_figure(spec)
        figure.patch.set_facecolor("#FFFFFF")
        axis.set_facecolor("#FFFFFF")
        axis.set_axisbelow(True)
        base_color = "#1D4ED8"
        deep_blue = "#1E3A8A"

        if spec.chart_type == "line" and spec.labels and spec.values:
            x_positions = list(range(len(spec.labels)))
            axis.plot(x_positions, spec.values, color=base_color, linewidth=4.0, marker="o", markersize=11)
            axis.fill_between(x_positions, spec.values, color=base_color, alpha=0.10)
            axis.set_xticks(x_positions)
            axis.set_xticklabels(spec.labels, fontproperties=bold_font, rotation=0)
            self._annotate_points(axis, x_positions, spec.values, number_font)

        elif spec.chart_type == "bar" and spec.labels and spec.values:
            x_positions = list(range(len(spec.labels)))
            bars = axis.bar(x_positions, spec.values, color=self._sequential_colors(len(spec.values)), width=0.62)
            axis.set_xticks(x_positions)
            axis.set_xticklabels(spec.labels, fontproperties=bold_font, rotation=25, ha="right")
            axis.bar_label(bars, labels=[f"{value:g}" for value in spec.values], padding=4, fontproperties=number_font, fontsize=22)

        elif spec.chart_type == "horizontal_bar" and spec.labels and spec.values:
            y_positions = list(range(len(spec.labels)))
            is_intensity_chart = spec.chart_key.startswith("intensity_")
            colors = self._sequential_colors(len(spec.values), reverse=True)
            compact_chart = spec.chart_key == "disaster_frequency" or is_intensity_chart
            label_size = 9 if is_intensity_chart else (10 if compact_chart else 17)
            number_size = 9 if is_intensity_chart else (10 if compact_chart else 22)
            bar_height = 0.34 if is_intensity_chart else (0.30 if compact_chart else 0.58)
            bars = axis.barh(y_positions, spec.values, color=colors, height=bar_height)
            axis.set_yticks(y_positions)
            axis.set_yticklabels(
                spec.labels,
                fontproperties=progress_font if is_intensity_chart else bold_font,
                fontsize=label_size,
                color="#111827",
            )
            axis.tick_params(axis="y", length=0, pad=8 if is_intensity_chart else 7)
            axis.invert_yaxis()
            if is_intensity_chart:
                # 指标图统一以 0 为基准，负温向左、正值向右，和用户给定的标准水平柱状图保持一致。
                minimum = min(float(value) for value in spec.values)
                maximum = max(float(value) for value in spec.values)
                span = max(maximum - minimum, abs(maximum), abs(minimum), 1.0)
                if minimum < 0 <= maximum:
                    axis.set_xlim(minimum - span * 0.10, maximum + span * 0.08)
                elif maximum <= 0:
                    axis.set_xlim(minimum - span * 0.10, 0)
                else:
                    axis.set_xlim(0, maximum + span * 0.12)
                axis.axvline(0, color="#D1D5DB", linewidth=0.9, zorder=0)
                label_pad = span * 0.018
                for value, y_pos in zip(spec.values, y_positions):
                    x_value = float(value)
                    if x_value < 0:
                        axis.text(
                            x_value - label_pad,
                            y_pos,
                            f"{x_value:g}",
                            va="center",
                            ha="right",
                            fontproperties=progress_number_font,
                            fontsize=number_size,
                            color="#000000",
                        )
                    else:
                        axis.text(
                            x_value + label_pad,
                            y_pos,
                            f"{x_value:g}",
                            va="center",
                            ha="left",
                            fontproperties=progress_number_font,
                            fontsize=number_size,
                            color="#000000",
                        )
            else:
                axis.bar_label(
                    bars,
                    labels=[f"{value:g}" for value in spec.values],
                    padding=4,
                    fontproperties=number_font,
                    fontsize=number_size,
                    color="#000000",
                )
        elif spec.chart_type == "progress_bar" and spec.labels and spec.values:
            max_value = max(spec.values) if spec.values else 1
            axis.set_xlim(0, 1)
            axis.set_ylim(-0.45, len(spec.labels) - 0.55)
            axis.axis("off")
            track_left, track_right = 0.20, 0.88
            for row_index, (label, value) in enumerate(zip(spec.labels, spec.values)):
                y_pos = len(spec.labels) - row_index - 1
                ratio = float(value) / max(max_value, 1)
                axis.text(0.035, y_pos, label, va="center", ha="left", fontproperties=progress_font, fontsize=12, color="#1F2937")
                axis.plot([track_left, track_right], [y_pos, y_pos], color="#E8EEF6", linewidth=3, solid_capstyle="round")
                axis.plot([track_left, track_left + (track_right - track_left) * ratio], [y_pos, y_pos], color="#2F7DEB", linewidth=3, solid_capstyle="round")
                axis.text(0.94, y_pos, f"{value:g}", va="center", ha="right", fontproperties=progress_number_font, fontsize=12, color="#000000")

        elif spec.chart_type == "lollipop" and spec.labels and spec.values:
            y_positions = list(range(len(spec.labels)))
            axis.hlines(y_positions, [0] * len(spec.values), spec.values, color="#93C5FD", linewidth=5)
            axis.scatter(spec.values, y_positions, s=170, color=base_color, edgecolor=deep_blue, linewidth=1.8, zorder=3)
            axis.set_yticks(y_positions)
            axis.set_yticklabels(spec.labels, fontproperties=bold_font)
            axis.invert_yaxis()
            for value, y_pos in zip(spec.values, y_positions):
                axis.text(value, y_pos, f"  {value:g}", va="center", fontproperties=number_font, fontsize=17, color="#000000")

        elif spec.chart_type == "heatmap" and spec.matrix:
            from matplotlib.colors import LinearSegmentedColormap

            matrix = np.array(spec.matrix, dtype=float)
            maximum = float(matrix.max()) if matrix.size else 0.0
            # 使用浅蓝到深蓝的连续色阶，零共现也保留极浅底色，不再形成大片空白单元格。
            cmap = LinearSegmentedColormap.from_list(
                "case_cooccurrence_blues",
                ["#F4F9FD", "#D9EBF7", "#7DB7DA", "#1F6FAF", "#0B3B6F"],
            )
            image = axis.imshow(
                matrix,
                cmap=cmap,
                aspect="equal",
                vmin=0,
                vmax=max(maximum, 1.0),
            )
            row_labels = spec.row_labels or spec.labels
            col_labels = spec.col_labels or row_labels
            axis.set_xticks(range(len(col_labels)))
            axis.set_xticklabels(col_labels, fontproperties=bold_font, fontsize=11, rotation=0, ha="center")
            axis.set_yticks(range(len(row_labels)))
            axis.set_yticklabels(row_labels, fontproperties=bold_font, fontsize=11)
            threshold = maximum * 0.58 if maximum else 0
            # 热力图单元格直接标注真实共现次数；零值也明确标出，避免误以为数据缺失。
            for row_index in range(matrix.shape[0]):
                for col_index in range(matrix.shape[1]):
                    value = matrix[row_index, col_index]
                    color = "#FFFFFF" if value > threshold else "#111827"
                    axis.text(
                        col_index,
                        row_index,
                        f"{value:g}",
                        ha="center",
                        va="center",
                        fontproperties=number_font,
                        fontsize=11,
                        color=color,
                    )
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label("共现次数", fontproperties=bold_font, fontsize=11, labelpad=8)
            for label in colorbar.ax.get_yticklabels():
                label.set_fontproperties(number_font)
                label.set_fontsize(10)
        elif spec.chart_type == "pie" and spec.labels and spec.values:
            colors = self._sequential_colors(len(spec.values))
            wedges, texts, autotexts = axis.pie(
                spec.values,
                labels=spec.labels,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors,
                pctdistance=0.60,
                labeldistance=1.08,
                textprops={"fontproperties": bold_font, "fontsize": 18, "fontweight": "bold"},
            )
            for autotext in autotexts:
                autotext.set_fontproperties(number_font)
                autotext.set_fontsize(14)
                autotext.set_color("#0F172A")
            axis.axis("equal")

        elif spec.chart_type == "histogram" and spec.values:
            axis.hist(spec.values, bins=min(8, max(3, len(spec.values))), color=base_color, alpha=0.82, edgecolor="#FFFFFF", linewidth=1.5)

        elif spec.chart_type == "boxplot" and spec.values:
            box = axis.boxplot(spec.values, vert=True, patch_artist=True, widths=0.42)
            for patch in box["boxes"]:
                patch.set_facecolor("#BFDBFE")
                patch.set_edgecolor(deep_blue)
                patch.set_linewidth(2.0)
            for key in ("whiskers", "caps", "medians"):
                for artist in box[key]:
                    artist.set_color(deep_blue)
                    artist.set_linewidth(2.0)
            axis.set_xticks([1])
            axis.set_xticklabels([spec.x_label or "指标"], fontproperties=bold_font)

        elif spec.chart_type == "scatter" and spec.x_values and spec.y_values:
            axis.scatter(spec.x_values, spec.y_values, s=190, color=base_color, edgecolor=deep_blue, linewidth=1.8, alpha=0.88)
            for label, x_value, y_value in zip(spec.labels, spec.x_values, spec.y_values):
                axis.annotate(label, (x_value, y_value), xytext=(7, 7), textcoords="offset points", fontproperties=bold_font, fontsize=21, color="#0F172A")

        compact_chart = spec.chart_key in {"disaster_frequency", "disaster_cooccurrence"} or spec.chart_key.startswith("intensity_")
        if spec.chart_type == "progress_bar":
            axis.set_title(spec.title, fontproperties=progress_font, fontsize=13, pad=6, color="#0F172A")
        elif spec.chart_key.startswith("intensity_"):
            axis.set_title(spec.title, fontproperties=bold_font, fontsize=13, pad=8, color="#0F172A")
        elif compact_chart:
            axis.set_title(spec.title, fontproperties=bold_font, fontsize=16, pad=10, color="#0F172A")
        else:
            axis.set_title(spec.title, fontproperties=bold_font, pad=22, color="#0F172A")
        if spec.x_label and spec.chart_type not in {"pie", "boxplot", "progress_bar"}:
            axis.set_xlabel(
                spec.x_label,
                fontproperties=bold_font,
                fontsize=9 if spec.chart_key.startswith("intensity_") else (12 if compact_chart else None),
                labelpad=7 if spec.chart_key.startswith("intensity_") else (10 if compact_chart else 16),
            )
        if spec.y_label and spec.chart_type not in {"pie", "progress_bar"}:
            axis.set_ylabel(
                spec.y_label,
                fontproperties=bold_font,
                fontsize=12 if compact_chart else None,
                labelpad=10 if compact_chart else 16,
            )
        if spec.chart_type == "pie":
            # 饼图不显示网格和边框，保持白底版面干净。
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
        elif spec.chart_type == "progress_bar":
            axis.grid(False)
        elif spec.chart_type == "heatmap":
            axis.grid(False)
            # 用白色细分隔线划分单元格，避免零值区域看起来像缺失或大片空白。
            axis.set_xticks([index - 0.5 for index in range(len(spec.col_labels or spec.labels) + 1)], minor=True)
            axis.set_yticks([index - 0.5 for index in range(len(spec.row_labels or spec.labels) + 1)], minor=True)
            axis.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.1)
            axis.tick_params(which="minor", bottom=False, left=False)
            for spine in axis.spines.values():
                spine.set_visible(False)
        else:
            axis.grid(True, axis="x" if spec.chart_type == "horizontal_bar" else "both", linestyle="--", linewidth=0.8, color="#CBD5E1", alpha=0.72)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#94A3B8")
            axis.spines["bottom"].set_color("#94A3B8")
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontproperties(bold_font if not label.get_text().replace(".", "", 1).isdigit() else number_font)
        figure.tight_layout(pad=1.25)
        figure.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="#FFFFFF")
        plt.close(figure)
        return output_path

    def _create_figure(self, spec: ChartSpec):
        """根据图表类型设置紧凑画布尺寸，避免插入 PDF 后文字被过度缩小。"""
        import matplotlib.pyplot as plt

        if spec.chart_type == "pie":
            return plt.subplots(figsize=(6.4, 4.8), dpi=420, facecolor="#FFFFFF")
        if spec.chart_type == "horizontal_bar":
            if spec.chart_key.startswith("intensity_"):
                height = max(2.6, 0.42 * max(len(spec.labels), 1) + 0.95)
                return plt.subplots(figsize=(7.2, height), dpi=420, facecolor="#FFFFFF")
            height = max(2.9, 0.32 * max(len(spec.labels), 1) + 1.20)
            return plt.subplots(figsize=(7.6, height), dpi=420, facecolor="#FFFFFF")
        if spec.chart_type == "progress_bar":
            height = max(1.05, 0.14 * max(len(spec.labels), 1) + 0.48)
            return plt.subplots(figsize=(7.0, height), dpi=420, facecolor="#FFFFFF")
        if spec.chart_type == "lollipop":
            height = max(4.5, 0.42 * max(len(spec.labels), 1) + 1.8)
            return plt.subplots(figsize=(7.0, height), dpi=420, facecolor="#FFFFFF")
        if spec.chart_type == "heatmap":
            size = max(len(spec.row_labels), len(spec.col_labels), len(spec.labels), 2)
            height = max(4.8, 0.56 * size + 1.75)
            return plt.subplots(figsize=(8.0, height), dpi=420, facecolor="#FFFFFF")
        if spec.chart_type == "boxplot":
            return plt.subplots(figsize=(6.6, 4.6), dpi=420, facecolor="#FFFFFF")
        if spec.chart_type == "histogram":
            return plt.subplots(figsize=(6.6, 4.6), dpi=420, facecolor="#FFFFFF")
        if spec.chart_type == "scatter":
            return plt.subplots(figsize=(6.8, 4.8), dpi=420, facecolor="#FFFFFF")
        return plt.subplots(figsize=(6.8, 4.8), dpi=420, facecolor="#FFFFFF")

    def _annotate_points(self, axis, x_values: list[float], y_values: list[float], number_font) -> None:
        """给折线图节点补充数值标注。"""
        for x_value, y_value in zip(x_values, y_values):
            axis.text(x_value, y_value, f"{y_value:g}", ha="center", va="bottom", fontproperties=number_font, fontsize=18, color="#1E3A8A")

    def _strongest_pair(self, labels: list[str], matrix: list[list[float]]) -> tuple[str, str, float] | None:
        """从热力图矩阵中找出最强的非对角共现关系。"""
        best: tuple[str, str, float] | None = None
        for row_index, row_label in enumerate(labels):
            if row_index >= len(matrix):
                break
            for col_index, col_label in enumerate(labels):
                if col_index >= len(matrix[row_index]) or row_index >= col_index:
                    continue
                value = float(matrix[row_index][col_index])
                if best is None or value > best[2]:
                    best = (row_label, col_label, value)
        return best

    def _sequential_colors(self, count: int, reverse: bool = False) -> list:
        """生成单色系渐变，避免图表出现太多跳色。"""
        import matplotlib.pyplot as plt

        if count <= 0:
            return []
        cmap = plt.get_cmap("Blues")
        if count == 1:
            return [cmap(0.72)]
        values = [0.30 + 0.52 * index / (count - 1) for index in range(count)]
        if reverse:
            values.reverse()
        return [cmap(value) for value in values]

    def _median(self, values: list[float]) -> float:
        """计算中位数，用于箱线图解读。"""
        ordered = sorted(values)
        length = len(ordered)
        middle = length // 2
        if length % 2 == 1:
            return float(ordered[middle])
        return float((ordered[middle - 1] + ordered[middle]) / 2)

    def _font_path(self) -> Path | None:
        """按优先级查找 Windows 和 Linux 中可用的中文字体。"""
        candidates = [
            # Windows 中文字体
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("C:/Windows/Fonts/simsun.ttf"),
            Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            # Ubuntu 安装 fonts-noto-cjk 后的中文字体
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
            # 兼容其他 Linux 发行版的常见字体目录
            Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ]
        return next((path for path in candidates if path.is_file()), None)

    def _times_font_path(self) -> Path | None:
        """查找 Times New Roman 字体，供纯数字标注和色标刻度使用。"""
        candidates = [
            Path("C:/Windows/Fonts/timesbd.ttf"),
            Path("C:/Windows/Fonts/times.ttf"),
            Path("C:/Windows/Fonts/timesbi.ttf"),
        ]
        return next((path for path in candidates if path.exists()), None)










