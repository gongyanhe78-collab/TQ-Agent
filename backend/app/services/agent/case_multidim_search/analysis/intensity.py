"""从原始文本提取可追溯的关键强度指标。"""
from __future__ import annotations

import re

from backend.app.models import DocumentChunk, StandardCase
from backend.app.services.agent.case_multidim_search.schemas import IntensityMetric


class IntensityExtractor:
    """只提取原文中明确出现的数值，不根据灾种推测强度。"""

    PATTERNS = (
        ('\u8fc7\u7a0b\u6700\u5927\u964d\u6c34\u91cf', '(?:\u8fc7\u7a0b\u6700\u5927\u964d\u6c34\u91cf|\u8fc7\u7a0b\u964d\u6c34\u91cf|\u7d2f\u8ba1\u964d\u6c34\u91cf)\\s*[:\uff1a]?\\s*(?:\u4e3a|\u8fbe|\u8fbe\u5230|\u53ef\u8fbe)?\\s*(\\d+(?:\\.\\d+)?)\\s*(mm|\u6beb\u7c73)'),
        ('\u6700\u5927\u5c0f\u65f6\u96e8\u5f3a', '(?:\u6700\u5927\u5c0f\u65f6\u96e8\u5f3a|\u5c0f\u65f6\u96e8\u5f3a|\u5c0f\u65f6\u6700\u5927\u96e8\u91cf|\u6700\u5927\u5c0f\u65f6\u964d\u6c34\u91cf|1\\s*\u5c0f\u65f6\u964d\u6c34\u91cf|1\\s*\u5c0f\u65f6\u6700\u5927\u964d\u6c34\u91cf)\\s*[:\uff1a]?\\s*(?:\u4e3a|\u8fbe|\u8fbe\u5230|\u53ef\u8fbe)?\\s*(\\d+(?:\\.\\d+)?)\\s*(mm/h|mm\\s*/\\s*h|\u6beb\u7c73/\u5c0f\u65f6|\u6beb\u7c73\u6bcf\u5c0f\u65f6|mm|\u6beb\u7c73)'),
        ('\u6700\u5927\u79ef\u96ea\u6df1\u5ea6', '(?:\u6700\u5927\u79ef\u96ea\u6df1\u5ea6|\u79ef\u96ea\u6df1\u5ea6|\u96ea\u6df1)\\s*[:\uff1a]?\\s*(?:\u4e3a|\u8fbe|\u8fbe\u5230|\u53ef\u8fbe)?\\s*(\\d+(?:\\.\\d+)?)\\s*(cm|\u5398\u7c73)'),
        ('\u6781\u5927\u98ce\u901f', '(?:\u6781\u5927\u98ce\u901f|\u6700\u5927\u9635\u98ce\u98ce\u901f|\u6700\u5927\u77ac\u65f6\u98ce\u901f|\u9635\u98ce\u98ce\u901f)\\s*[:\uff1a]?\\s*(?:\u4e3a|\u8fbe|\u8fbe\u5230|\u53ef\u8fbe)?\\s*(\\d+(?:\\.\\d+)?)\\s*(m\\s*/\\s*s|m/s|m\xb7s-1|m\xb7s\\s*-\\s*1|\u7c73/\u79d2|\u7c73\u6bcf\u79d2)'),
        ('\u6700\u5927\u98ce\u901f', '(?:\u6700\u5927\u98ce\u901f|\u6700\u5927\u5e73\u5747\u98ce\u901f)\\s*[:\uff1a]?\\s*(?:\u4e3a|\u8fbe|\u8fbe\u5230|\u53ef\u8fbe)?\\s*(\\d+(?:\\.\\d+)?)\\s*(m\\s*/\\s*s|m/s|m\xb7s-1|m\xb7s\\s*-\\s*1|\u7c73/\u79d2|\u7c73\u6bcf\u79d2)'),
        ('\u9635\u98ce\u98ce\u529b', '(?:\u9635\u98ce\u98ce\u529b|\u6700\u5927\u98ce\u529b\u7b49\u7ea7|\u9635\u98ce|\u98ce\u529b)\\s*[:\uff1a]?\\s*(?:\u4e3a|\u8fbe|\u8fbe\u5230|\u53ef\u8fbe)?\\s*(\\d+(?:\\.\\d+)?)\\s*(\u7ea7|\u7ea7\u4ee5\u4e0a)'),
        ('\u6700\u9ad8\u6c14\u6e29', '(?:\u6700\u9ad8\u6c14\u6e29|\u65e5\u6700\u9ad8\u6c14\u6e29|\u6700\u9ad8\u6e29\u5ea6)\\s*[:\uff1a]?\\s*(?:\u4e3a|\u8fbe|\u8fbe\u5230|\u5347\u81f3|\u8d85\u8fc7)?\\s*(?<![-\uff0d])((?:2[5-9]|[3-4]\\d|5[0-5])(?:\\.\\d+)?)\\s*(\u2103|\xb0C|C)'),
        ('\u6700\u4f4e\u6c14\u6e29', '(?:\u6700\u4f4e\u6c14\u6e29|\u65e5\u6700\u4f4e\u6c14\u6e29|\u6700\u4f4e\u6e29\u5ea6)\\s*[:\uff1a]?\\s*(?:\u4e3a|\u8fbe|\u8fbe\u5230|\u964d\u81f3)?\\s*(-?\\d+(?:\\.\\d+)?)\\s*(\u2103|\xb0C|C)'),
        ('\u8fc7\u7a0b\u964d\u6e29\u5e45\u5ea6', '(?:\u8fc7\u7a0b\u964d\u6e29\u5e45\u5ea6|\u964d\u6e29\u5e45\u5ea6|\u6700\u5927\u964d\u6e29)\\s*[:\uff1a]?\\s*(?:\u4e3a|\u8fbe|\u8fbe\u5230|\u8d85\u8fc7)?\\s*(\\d+(?:\\.\\d+)?)\\s*(\u2103|\xb0C|C)'),
        ("过程最大降水量", r"(?:过程|累计|24\s*h|24\s*小时|48\s*小时)[^。；]{0,80}?(?:最大(?:值|降水量|降水)?(?:出现在|为|达到|达)?|最大(?:出现在)?)[^。；]{0,30}?(\d+(?:\.\d+)?)\s*(mm|毫米)"),
        ("最大小时雨强", r"(?:最大小时雨强|最大雨强|小时最大雨量|小时最大降水量|最大小时降水量|最大1\s*小时降水量|1\s*小时最大降水量|小时雪强|小时雨强|小时雨量|１小时降水量|1\s*小时降水量|每小时雨强)[^。；]{0,80}?(?:为|达到|达|介于|可达)?\s*(\d+(?:\.\d+)?)\s*(mm/h|mm\s*/\s*h|毫米/小时|毫米每小时|mm|毫米)"),
        ("最大降水量", r"(?:(?:中心点|中心附近|中心)?|小时|过程|国家站|区域站)?最大(?:降水量|降水|雨强|雪强)\s*[:：]?\s*(?:为|达到|达|出现在|可达)?[^。；]{0,40}?(\d+(?:\.\d+)?)\s*(mm|毫米|mm/h|毫米/小时)"),
        ("极大风速", r"(?:极大风速|最大阵风风速|最大瞬时风速|瞬时大风|日极端阵风|阵风风速)[^。；]{0,100}?(?:为|达到|达|出现在|可达)?[^。；]{0,40}?(\d+(?:\.\d+)?)\s*(m\s*/\s*s|m/s|m·s-1|m·s\s*-\s*1|米/秒|米每秒)"),
        ("最大风速", r"(?:(?:中心点|中心附近|中心)?最大风速\s*[:：]?\s*(?:为|达到|达|可达)?|(?:最大(?:值|风速)?(?:出现在|为|达到|达|可达)?|最强风力)[^。；]{0,70}?|风力超过[^。；]{0,120}?[（(])(\d+(?:\.\d+)?)\s*(m\s*/\s*s|m/s|m·s-1|m·s\s*-\s*1|米/秒|米每秒)"),
        ("阵风风力", r"(?:阵风|风力|阵风风力|最大风力等级)[^。；]{0,60}?(?:达|达到|可达|为|有)?\s*(\d+(?:\.\d+)?)\s*(级|级以上)"),
        ("最高气温", r"(?:最高气温|日最高气温|最高温度)[^。；]{0,80}?(?:为|达到|达|升至|超过|在)?[^。；]{0,30}?(?<![-－])((?:2[5-9]|[3-4]\d|5[0-5])(?:\.\d+)?)\s*(℃|°C|C)"),
        ("最低气温", r"(?:最低气温|日最低气温|最低温度)[^。；]{0,80}?(?:为|达到|达|降至|出现在)?[^。；]{0,30}?(-?\d+(?:\.\d+)?)\s*(℃|°C|C)"),
        ("过程降温幅度", r"(?:降温幅度|最低气温[^。；]{0,20}?下降|气温[^。；]{0,20}?下降|降温最大值)[^。；]{0,80}?(?:为|达到|达|超过|超|介于)?[^。；]{0,20}?(\d+(?:\.\d+)?)\s*(℃|°C|C)"),
        ("最大积雪深度", r"(?:积雪深度|雪深)[^。；]{0,80}?(?:最大(?:为|出现在)?|达|达到|介于)?[^。；]{0,20}?(\d+(?:\.\d+)?)\s*(cm|厘米)"),
        ("过程最大降雪量", r"(?:降雪量|过程降雪|预警区域内降雪量)[^。；]{0,80}?(?:最大(?:为|出现在)?|达|达到|将达)?[^。；]{0,20}?(\d+(?:\.\d+)?)\s*(mm|毫米)"),
        ("最低能见度", r"(?:最低能见度|最小水平能见度)[^。；]{0,60}?(?:为|达到|达|降至)?\s*(\d+(?:\.\d+)?)\s*(km|公里|千米|m(?!\s*/\s*s)|米)"),
        ("最大冰雹直径", r"(?:冰雹直径|最大冰雹|冰雹)[^。；]{0,80}?(?:为|达到|达)?[^。；]{0,20}?(\d+(?:\.\d+)?)\s*(mm|毫米|cm|厘米)"),
        ("雷达回波强度", r"(?:中心点|中心附近|中心)?(?:雷达|组合)?回波(?:强度|反射率因子)?\s*[:：]?\s*(?:达|达到|为|在)?\s*(\d+(?:\.\d+)?)\s*(dBZ|DBZ|dBz)"),
        ("中心气压", r"中心(?:点)?气压\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(hPa|百帕)"),
        ("中心温度", r"中心(?:点)?温度\s*[:：]?\s*(-?\d+(?:\.\d+)?)\s*(℃|°C|C)"),
    )
    def extract(self, case: StandardCase, chunks: list[DocumentChunk]) -> list[IntensityMetric]:
        """从个例原文中提取全部有证据的强度指标。"""
        texts: list[tuple[str, str]] = []
        for chunk in chunks:
            chunk_id = str(chunk.chunk_id or "")
            # 强度汇总只能使用标准个例显式关联的正文，不能混入图片图注或附近文字。
            if not case.source_chunk_ids or chunk_id in case.source_chunk_ids:
                texts.append((chunk_id, chunk.content))
        # 强度汇总表属于核心证据，只从个例关联正文抽取，避免标准化摘要串入错误单位。
        if not texts:
            return []

        found: list[tuple[int, IntensityMetric]] = []
        for source_chunk_id, text in texts:
            for metric_name, pattern in self.PATTERNS:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    value = float(match.group(1))
                    unit = self._normalize_unit(match.group(2))
                    matched_text = match.group(0)
                    # “小时最大降水量”属于小时雨强，不能再作为过程累计降水量重复入表。
                    if metric_name in {"最大降水量", "过程最大降水量"} and re.search(
                        r"(?:最大小时|小时最大|小时雨强|每小时雨强)", matched_text
                    ):
                        continue
                    # “1小时降水量达30mm”语义上是小时雨强，统一换算为 mm/h 便于统计和绘图。
                    if metric_name == "最大小时雨强" and unit == "mm":
                        unit = "mm/h"
                    excerpt = self._excerpt(text, match.start(), match.end())
                    value, unit = self._normalize_metric_value(metric_name, value, unit, excerpt)
                    if not self._unit_matches_metric(metric_name, unit):
                        continue
                    if self._should_skip_metric(metric_name, value, unit, excerpt):
                        continue
                    found.append(
                        (
                            match.start(),
                            IntensityMetric(
                                metric_name=metric_name,
                                value=value,
                                unit=unit,
                                location=self._location(excerpt),
                                relation="中心点" if "中心" in excerpt else "",
                                source_chunk_id=source_chunk_id,
                                source_text=excerpt,
                                confidence=0.95 if "中心" in excerpt else 0.8,
                            ),
                        )
                    )
            found.extend(self._supplemental_metrics(source_chunk_id, text))

        deduped: list[IntensityMetric] = []
        seen_indexes: dict[tuple, int] = {}
        relation_priority = {"范围上限": 0, "原文提取": 1, "站点实测": 2}
        for _, metric in sorted(found, key=lambda item: item[0]):
            key = (metric.metric_name, metric.value, metric.unit)
            if key not in seen_indexes:
                seen_indexes[key] = len(deduped)
                deduped.append(metric)
                continue
            # 同一数值同时来自范围和站点清单时，保留可核验性更高的具体实测证据。
            index = seen_indexes[key]
            previous = deduped[index]
            if relation_priority.get(metric.relation, 1) > relation_priority.get(previous.relation, 1):
                deduped[index] = metric
        return deduped




    def _supplemental_metrics(self, source_chunk_id: str, text: str) -> list[tuple[int, IntensityMetric]]:
        """补充抽取业务原文里的常见自然句式，避免强度汇总过度依赖大模型。"""
        found: list[tuple[int, IntensityMetric]] = []
        for sentence, offset in self._sentences_with_offset(text):
            candidates = [
                # 业务材料常写“24小时累计降水量 0.1-26.6mm”，省略“为/介于”也必须取范围上限。
                ("过程最大降水量", "mm", r"(?:(?:\d{1,2}\s*(?:小时|h))\s*)?(?:累计|过程)?降水量[^。；]{0,40}?(?:为|介于)?\s*\d+(?:\.\d+)?\s*[～~\-－—至到]+\s*(\d+(?:\.\d+)?)\s*(mm|毫米)"),
                ("过程最大降水量", "mm", r"降水量[^。；]{0,80}?介于\s*\d+(?:\.\d+)?\s*[～~\-－—至到]+\s*(\d+(?:\.\d+)?)\s*(mm|毫米)"),
                # 降水范围右端是该过程可追溯的最大值，例如“累计降水量为 0.1～150.4 毫米”。
                ("过程最大降水量", "mm", r"(?:累计|过程)?降水量[^。；]{0,80}?(?:介于|为)\s*\d+(?:\.\d+)?\s*[～~\-－—至到]+\s*(\d+(?:\.\d+)?)\s*(mm|毫米)"),
                # 过程极值可能先给区域站、后给国家站，必须识别“过程累计降水量最高达303mm”。
                ("过程最大降水量", "mm", r"(?:过程)?累计降水量[^。；]{0,80}?(?:最高|最大)(?:值)?\s*(?:达到|达|为)?\s*(\d+(?:\.\d+)?)\s*(mm|毫米)"),
                ("过程最大降水量", "mm", r"(?:暴雨|降水)[^。；]{0,100}?最大[^。；]{0,50}?(\d+(?:\.\d+)?)\s*(mm|毫米)"),
                # 高温实况常写成范围，必须取右端而不能使用“超过 40℃”等门槛值。
                ("最高气温", "℃", r"(?:最高气温|日最高气温|最高温度)[^。；]{0,80}?(?:介于|为)\s*\d+(?:\.\d+)?\s*[～~\-－—至到]+\s*(\d+(?:\.\d+)?)\s*(℃|°C|C)"),
                # “积雪深度≧5cm 的有 8 站，最大为10cm”要使用后面的最大值。
                ("最大积雪深度", "cm", r"(?:积雪深度|雪深)[^。；]{0,120}?最大(?:为|达|达到|出现在)?\s*(\d+(?:\.\d+)?)\s*(cm|厘米)"),
                # 雨强只给范围时取上限；若另有国家站、区域站实测极值，汇总阶段优先实测值。
                ("最大小时雨强", "mm/h", r"(?:雨强|小时雨量|短时强降水)[^。；]{0,100}?(?:介于|为)\s*\d+(?:\.\d+)?\s*[～~\-－—至到]+\s*(\d+(?:\.\d+)?)\s*(mm/h|mm\s*/\s*h|毫米/小时|mm|毫米)"),
                ("最大小时雨强", "mm/h", r"(?:雨强|短时强降水)[^。；]{0,140}?最大[^。；]{0,50}?(\d+(?:\.\d+)?)\s*(mm/h|mm\s*/\s*h|毫米/小时|mm|毫米)"),
                ("最大小时雨强", "mm/h", r"(?:国家站|区域站|自动站)?[^。；]{0,30}?最大[^。；]{0,50}?(\d+(?:\.\d+)?)\s*(mm/h|mm\s*/\s*h|毫米/小时)"),
                # PDF 换行清理后可稳定识别“最大值出现在某站，达到27.4m/s”句式。
                ("最大风速", "m/s", r"最大值[^。；]{0,80}?(?:达到|达|为)\s*(\d+(?:\.\d+)?)\s*(m\s*/\s*s|m/s|米/秒|米每秒)"),
                # 降温幅度范围取右端，例如“降温介于1.4-13.3℃之间”。
                ("过程降温幅度", "℃", r"(?:降温幅度|最低气温[^。；]{0,20}?降温|气温[^。；]{0,20}?下降|降温)[^。；]{0,80}?(?:介于|为)\s*\d+(?:\.\d+)?\s*[～~\-－—至到]+\s*(\d+(?:\.\d+)?)\s*(℃|°C|C)"),
                ("阵风风力", "级", r"(?:大风|阵风|风力)[^。；]{0,140}?(?:最大值|最大|达到|达)[^。；]{0,50}?(\d+(?:\.\d+)?)\s*(级|级以上)"),
                # “局地出现8-10级雷暴大风”取等级范围上限10级。
                ("阵风风力", "级", r"(?:出现|有)[^。；]{0,40}?\d+(?:\.\d+)?\s*[～~\-－—至到]+\s*(\d+(?:\.\d+)?)\s*(级)(?:以上)?(?:的)?(?:雷暴)?大风"),
            ]
            for metric_name, expected_unit, pattern in candidates:
                # “中心点最大降水量”是单点指标，不能同时伪装成过程最大降水量。
                if metric_name == "过程最大降水量" and "中心点" in sentence:
                    continue
                for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
                    value = float(match.group(1))
                    unit = self._normalize_unit(match.group(2))
                    if metric_name == "最大小时雨强" and unit == "mm":
                        unit = "mm/h"
                    if unit != expected_unit:
                        continue
                    excerpt = sentence.strip()[:240]
                    if self._should_skip_metric(metric_name, value, unit, excerpt):
                        continue
                    relation = "范围上限" if "介于" in pattern else "原文提取"
                    found.append((offset + match.start(), self._metric(metric_name, value, unit, source_chunk_id, excerpt, confidence=0.86, relation=relation)))
            found.extend(self._temperature_drop_station_metrics(source_chunk_id, sentence, offset))
            found.extend(self._visibility_metrics(source_chunk_id, sentence, offset))
        return found

    def _temperature_drop_station_metrics(self, source_chunk_id: str, sentence: str, offset: int) -> list[tuple[int, IntensityMetric]]:
        """从降温站点清单中提取实测幅度，解决站名后数值未重复书写单位的问题。"""
        if "降温" not in sentence or "站" not in sentence or "：" not in sentence:
            return []
        station_text = sentence.split("：", 1)[1]
        found: list[tuple[int, IntensityMetric]] = []
        pattern = r"[\u4e00-\u9fff]{1,10}\s*(-?\d{1,2}(?:\.\d+)?)"
        for match in re.finditer(pattern, station_text):
            value = float(match.group(1))
            if not 0 < value <= 30:
                continue
            found.append(
                (
                    offset + sentence.find("：") + 1 + match.start(),
                    self._metric(
                        "过程降温幅度",
                        value,
                        "℃",
                        source_chunk_id,
                        sentence.strip()[:240],
                        confidence=0.9,
                        relation="站点实测",
                    ),
                )
            )
        return found
    def _visibility_metrics(self, source_chunk_id: str, sentence: str, offset: int) -> list[tuple[int, IntensityMetric]]:
        """从能见度实况句中提取全部距离值，排除 m/s 风速，供后续选择最低值。"""
        if "能见度" not in sentence:
            return []
        found: list[tuple[int, IntensityMetric]] = []
        # 仅把独立距离单位当作能见度；m/s 不能被错误截成 m。
        pattern = r"(\d+(?:\.\d+)?)\s*(km|公里|千米|m(?!\s*/\s*s)|米)"
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            value = float(match.group(1))
            unit = self._normalize_unit(match.group(2))
            value, unit = self._normalize_metric_value("最低能见度", value, unit, sentence)
            if unit != "km" or self._should_skip_metric("最低能见度", value, unit, sentence):
                continue
            found.append((offset + match.start(), self._metric("最低能见度", value, unit, source_chunk_id, sentence.strip()[:240], confidence=0.88)))
        return found

    def _sentences_with_offset(self, text: str) -> list[tuple[str, int]]:
        """先修复 PDF 行内断行，再按句号和分号切分原文并保留起始位置。"""
        normalized = re.sub(r"\s*\r?\n\s*", "", str(text or ""))
        normalized = re.sub(r"[\t ]+", " ", normalized)
        parts: list[tuple[str, int]] = []
        start = 0
        for match in re.finditer(r"[。；]+", normalized):
            sentence = normalized[start:match.start()].strip()
            if sentence:
                parts.append((sentence, start))
            start = match.end()
        tail = normalized[start:].strip()
        if tail:
            parts.append((tail, start))
        return parts

    def _metric(        self,
        metric_name: str,
        value: float,
        unit: str,
        source_chunk_id: str,
        excerpt: str,
        confidence: float,
        relation: str = "原文提取",
    ) -> IntensityMetric:
        """统一构造强度指标，保证规则补抽也带有原文证据。"""
        return IntensityMetric(
            metric_name=metric_name,
            value=value,
            unit=unit,
            location=self._location(excerpt),
            relation=relation,
            source_chunk_id=source_chunk_id,
            source_text=excerpt,
            confidence=confidence,
        )

    def _unit_matches_metric(self, metric_name: str, unit: str) -> bool:
        """校验指标和单位是否匹配，防止不同强度字段之间发生串列。"""
        if metric_name in {'过程最大降水量', '最大降水量', '过程最大降雪量'}:
            return unit == "mm"
        if metric_name == '最大小时雨强':
            return unit == "mm/h"
        if metric_name in {'最高气温', '最低气温', '过程降温幅度', '中心温度'}:
            return unit == '℃'
        if metric_name in {'极大风速', '最大风速'}:
            return unit == "m/s"
        if metric_name == '阵风风力':
            return unit == '级'
        if metric_name == '最大积雪深度':
            return unit == "cm"
        if metric_name == '最低能见度':
            return unit == "km"
        if metric_name == '最大冰雹直径':
            return unit in {"mm", "cm"}
        if metric_name == '雷达回波强度':
            return unit == "dBZ"
        if metric_name == '中心气压':
            return unit == "hPa"
        return True

    def _normalize_metric_value(self, metric_name: str, value: float, unit: str, excerpt: str) -> tuple[float, str]:
        """根据指标语义做必要单位换算，避免同一图表混用不同单位。"""
        if metric_name == "最低能见度" and unit == "m":
            return value / 1000, "km"
        return value, unit

    def _should_skip_metric(self, metric_name: str, value: float, unit: str, excerpt: str) -> bool:
        """\u8fc7\u6ee4\u660e\u663e\u4e0d\u662f\u5f53\u524d\u4e2a\u4f8b\u5f3a\u5ea6\u7684\u6570\u503c\uff0c\u4f8b\u5982\u6708\u5ea6\u6c14\u5019\u6982\u51b5\u3001\u9ad8\u7a7a\u51b7\u4e2d\u5fc3\u6e29\u5ea6\u6216\u96f6\u503c\u5360\u4f4d\u3002"""
        positive_metrics = {
            "\u8fc7\u7a0b\u6700\u5927\u964d\u6c34\u91cf", "\u6700\u5927\u964d\u6c34\u91cf", "\u6700\u5927\u5c0f\u65f6\u96e8\u5f3a", "\u6700\u5927\u79ef\u96ea\u6df1\u5ea6",
            "\u8fc7\u7a0b\u6700\u5927\u964d\u96ea\u91cf", "\u6700\u5927\u98ce\u901f", "\u6781\u5927\u98ce\u901f", "\u9635\u98ce\u98ce\u529b",
        }
        if metric_name in positive_metrics and value <= 0:
            return True
        if metric_name in {"\u6700\u5927\u98ce\u901f", "\u6781\u5927\u98ce\u901f", "\u9635\u98ce\u98ce\u529b"} and self._is_wind_forecast_or_warning(excerpt):
            # 预警中的“阵风可达”是预报阈值，不能覆盖已发生过程的站点实测风速和风级。
            return True
        if metric_name in {"\u8fc7\u7a0b\u6700\u5927\u964d\u6c34\u91cf", "\u6700\u5927\u964d\u6c34\u91cf", "\u6700\u5927\u5c0f\u65f6\u96e8\u5f3a"} and self._is_monthly_precip_overview(excerpt):
            return True
        if metric_name in {"\u6700\u9ad8\u6c14\u6e29", "\u6700\u4f4e\u6c14\u6e29"} and any(word in excerpt for word in ("\u51b7\u4e2d\u5fc3", "\u4e2d\u5fc3\u6c14\u6e29", "500hPa", "700hPa", "850hPa")):
            return True
        if metric_name == "\u6700\u9ad8\u6c14\u6e29" and value < 30:
            return True
        if metric_name == "\u6700\u9ad8\u6c14\u6e29" and value > 55:
            return True
        if metric_name == "\u6700\u4f4e\u6c14\u6e29" and value >= 0:
            return True
        if metric_name == "\u8fc7\u7a0b\u964d\u6e29\u5e45\u5ea6" and (value <= 0 or "\u6700\u4f4e\u6c14\u6e29\u51fa\u73b0\u5728" in excerpt and value > 25):
            return True
        return False

    def _is_wind_forecast_or_warning(self, excerpt: str) -> bool:
        """识别大风预警、预报文字，防止“预计可达”被误作实况极值。"""
        text = re.sub(r"\s+", "", str(excerpt or ""))
        return bool(
            "\u9884\u8b66\u533a\u57df" in text
            or ("\u9884\u8b66" in text and any(token in text for token in ("\u9884\u8ba1", "\u53ef\u8fbe", "\u672a\u6765")))
            or ("\u9884\u8ba1" in text and any(token in text for token in ("\u9635\u98ce", "\u98ce\u529b", "\u98ce\u901f")))
        )

    def _is_monthly_precip_overview(self, excerpt: str) -> bool:
        """\u8bc6\u522b\u6708\u5ea6\u6c14\u8c61\u6982\u51b5\u4e2d\u7684\u5168\u7701\u964d\u6c34\u80cc\u666f\u503c\uff0c\u907f\u514d\u4e32\u5165\u5355\u4e2a\u5929\u6c14\u8fc7\u7a0b\u5f3a\u5ea6\u8868\u3002"""
        compact_text = re.sub(r"\s+", "", str(excerpt or ""))
        if not compact_text:
            return False
        if re.search(r"\d{4}\u5e74\d{1,2}\u6708", compact_text) and "\u5c71\u897f\u7701\u964d\u6c34\u91cf\u4ecb\u4e8e" in compact_text:
            return True
        overview_terms = ("\u6708\u964d\u6c34\u91cf", "\u5e73\u5747\u964d\u6c34\u91cf", "\u5168\u7701\u5e73\u5747", "\u8f83\u5e38\u5e74", "\u6c14\u5019\u6982\u51b5", "\u6c14\u8c61\u6982\u51b5")
        return "\u5c71\u897f\u7701\u964d\u6c34\u91cf" in compact_text and any(term in compact_text for term in overview_terms)

    def _normalize_unit(self, unit: str) -> str:
        """统一强度指标使用的业务单位。"""
        return {
            "米/秒": "m/s",
            "米每秒": "m/s",
            "m / s": "m/s",
            "m/s": "m/s",
            "m·s-1": "m/s",
            "m·s - 1": "m/s",
            "毫米": "mm",
            "毫米/小时": "mm/h",
            "毫米每小时": "mm/h",
            "mm / h": "mm/h",
            "DBZ": "dBZ",
            "dBz": "dBZ",
            "百帕": "hPa",
            "°C": "℃",
            "C": "℃",
            "厘米": "cm",
            "公里": "km",
            "千米": "km",
            "米": "m",
            "级以上": "级",
            "级": "级",
        }.get(unit, unit)

    def _excerpt(self, text: str, start: int, end: int) -> str:
        """截取强度数值所在的原文证据片段。"""
        left = max(0, text.rfind("。", 0, start) + 1)
        right = text.find("。", end)
        return text[left:(right + 1 if right >= 0 else len(text))].strip()[:240]

    def _location(self, excerpt: str) -> str:
        """从强度证据片段中识别影响地市。"""
        for city in ("太原", "大同", "朔州", "忻州", "阳泉", "晋中", "吕梁", "长治", "晋城", "临汾", "运城"):
            if city in excerpt:
                return city
        return ""






