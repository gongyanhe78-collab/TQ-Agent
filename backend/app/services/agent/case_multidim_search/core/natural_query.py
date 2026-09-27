"""自然语言多维检索条件解析、上下文合并和会话提案管理。"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from threading import Lock
from time import time
from typing import Any
from uuid import uuid4

from backend.app.services.agent.case_multidim_search.schemas import (
    NaturalCaseQueryParseResponse,
    StructuredCaseSearchRequest,
)


DISASTER_NAMES = (
    "雷暴大风", "短时强降水", "大暴雨", "强对流", "强降水", "暴雨", "暴雪", "雨雪",
    "冰雹", "寒潮", "低温", "高温", "沙尘", "霜冻", "降雪", "雷暴", "大风", "雾",
)
CITY_NAMES = ("太原", "大同", "朔州", "忻州", "阳泉", "晋中", "吕梁", "长治", "晋城", "临汾", "运城")
AREA_NAMES = ("晋北", "晋中", "晋南", "山西北部", "山西中部", "山西南部", "全省", "山西")
SEASON_MONTHS = {
    "春季": [3, 4, 5], "春天": [3, 4, 5],
    "夏季": [6, 7, 8], "夏天": [6, 7, 8],
    "秋季": [9, 10, 11], "秋天": [9, 10, 11],
    "冬季": [12, 1, 2], "冬天": [12, 1, 2],
}
QUERY_FIELDS = ("start_date", "end_date", "years", "months", "disaster_types", "cities", "areas")
OPERATION_LABELS = {
    "new": "按本轮消息新建检索，不继承上一轮条件",
    "restrict": "保留上一轮其他条件，并增加或更新本轮限定维度",
    "scope": "保留上一轮其他条件，只查看本轮指定范围",
    "replace": "保留上一轮其他条件，并替换本轮明确修改的维度",
}


@dataclass
class NaturalExtraction:
    """规则和模型从本轮消息中识别出的受控字段。"""

    values: dict[str, Any] = field(default_factory=dict)
    touched: set[str] = field(default_factory=set)
    clear: set[str] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    parse_status: str = "rule"


class NaturalCaseQueryParser:
    """把自然语言转换成现有 StructuredCaseSearchRequest。"""

    def __init__(self, llm_client=None):
        """保存可选大模型客户端；明确条件始终以确定性规则为准。"""
        self.llm_client = llm_client

    def parse(
        self,
        message: str,
        committed: StructuredCaseSearchRequest | None = None,
    ) -> tuple[str, StructuredCaseSearchRequest, list[str], list[str], list[str], str, bool]:
        """解析本轮消息，并结合已确认条件判断本轮是新检索还是上下文修改。"""
        text = re.sub(r"\s+", " ", str(message or "")).strip()
        extraction = self._rule_extract(text)
        self._supplement_with_llm(text, extraction)
        operation = self._operation(text, committed, extraction)

        warnings = list(extraction.warnings)
        if committed is None and operation != "new":
            warnings.append("当前会话还没有已确认的检索条件，无法执行上下文修改；请先完整描述一次检索。")
            empty = self._request_from_values(extraction.values)
            return operation, empty, self._display_lines(empty), [], warnings, extraction.parse_status, False

        resolved, changes = self._merge(operation, extraction, committed)
        can_confirm = self._has_filter(resolved) or bool(re.search(r"(?:全部|所有)个例", text))
        if not extraction.touched and not extraction.clear:
            can_confirm = False
            warnings.append("没有识别到可执行的时间、灾种、地市或区域条件，请补充至少一个明确条件。")
        if "北部" in text and not any(term in text for term in ("山西北部", "晋北")):
            warnings.append("“北部”没有自动映射区域，请明确写“山西北部”或“晋北”。")
            can_confirm = False
        return operation, resolved, self._display_lines(resolved), changes, warnings, extraction.parse_status, can_confirm

    def _operation(
        self,
        text: str,
        committed: StructuredCaseSearchRequest | None,
        extraction: NaturalExtraction,
    ) -> str:
        """以明确语义、模型判断和保守兜底共同判定上下文操作。"""
        if re.search(r"(?:重新|另查|新查|新检索|重新检索|重新查询|换一个查询)", text):
            return "new"
        if re.search(r"(?:再限定|再加上|增加条件|继续限定|补充条件)", text):
            return "restrict"
        if re.search(r"(?:只看|仅看|只检索|仅检索|范围缩小到|筛到)", text):
            return "scope"
        if re.search(r"(?:改成|改为|换成|调整为|替换成|改到)", text):
            return "replace"
        if committed is None:
            return "new"

        # 完整、自足的查询优先视为新检索，避免模型把普通新问题错误继承到历史条件。
        touched_count = len(extraction.touched)
        if re.search(r"(?:查询|检索|查找|统计|分析).*(?:过程|个例|天气|报告)", text) and touched_count >= 2:
            return "new"

        model_operation = self._classify_operation_with_llm(text, committed, extraction)
        if model_operation:
            extraction.parse_status = "rule_and_context_llm"
            return model_operation

        # 模型不可用时只处理语义十分明确的指代或承接表达，其余一律新建检索。
        if re.search(r"(?:其中|这些|上述|刚才|上一轮|前面|原条件|范围内|结果里|继续|另外|也要|还要)", text):
            if "disaster" in extraction.touched and touched_count == 1:
                return "scope"
            if "time" in extraction.touched and touched_count == 1:
                return "replace"
            return "restrict"
        return "new"

    def _classify_operation_with_llm(
        self,
        text: str,
        committed: StructuredCaseSearchRequest,
        extraction: NaturalExtraction,
    ) -> str:
        """让模型结合已确认条件判断上下文关系，仅接受高置信白名单结果。"""
        if self.llm_client is None:
            return ""
        try:
            if not self.llm_client.is_available():
                return ""
        except Exception:
            return ""
        prompt = (
            "判断用户本轮气象检索消息与上一轮已确认条件的关系，只输出JSON："
            '{"operation":"new|restrict|scope|replace","confidence":0到1,"reason":"简短理由"}。'
            "new表示独立的新检索；restrict表示在原条件上增加或收紧地区等过滤；"
            "scope表示只保留指定灾种或子范围；replace表示替换明确修改的时间或其他维度。"
            "不要依赖固定关键词，要理解指代、省略和上下文；如果无法确认用户想继承，必须输出new。"
        )
        context = {
            "上一轮已确认条件": {
                field: getattr(committed, field) for field in QUERY_FIELDS
            },
            "本轮消息": text,
            "本轮规则识别维度": sorted(extraction.touched),
            "本轮规则识别值": extraction.values,
        }
        try:
            raw = str(self.llm_client.answer_with_context(
                prompt,
                [json.dumps(context, ensure_ascii=False)],
                max_tokens=260,
            ) or "")
            payload = self._json_object(raw)
            operation = str(payload.get("operation") or "").strip().lower()
            confidence = float(payload.get("confidence") or 0)
        except (TypeError, ValueError, OSError):
            return ""
        return operation if operation in OPERATION_LABELS and confidence >= 0.72 else ""

    def _rule_extract(self, text: str) -> NaturalExtraction:
        """使用受控词表和日期规则提取明确条件。"""
        extraction = NaturalExtraction(values={field: [] if field not in {"start_date", "end_date"} else "" for field in QUERY_FIELDS})
        disasters = self._non_overlapping_terms(text, DISASTER_NAMES)
        cities = [name for name in CITY_NAMES if name in text]
        areas = [name for name in AREA_NAMES if name in text]
        # “山西”通常只是报告范围背景，不自动变成区域过滤；“晋中”优先解释为地市。
        if "山西" in areas:
            areas.remove("山西")
        if "晋中" in cities and "晋中" in areas:
            areas.remove("晋中")
        if disasters:
            extraction.values["disaster_types"] = disasters
            extraction.touched.add("disaster")
        if cities or areas:
            extraction.values["cities"] = cities
            extraction.values["areas"] = areas
            extraction.touched.add("region")

        temporal = self._extract_temporal(text)
        if temporal is not None:
            extraction.values.update(temporal)
            extraction.touched.add("time")

        if re.search(r"(?:不限|清空|取消)(?:时间|日期|月份|年份)", text):
            extraction.clear.add("time")
            extraction.touched.add("time")
        if re.search(r"(?:不限|清空|取消)(?:灾种|灾害类型)", text):
            extraction.clear.add("disaster")
            extraction.touched.add("disaster")
        if re.search(r"(?:(?:不限|清空|取消)(?:地区|区域|地市)|(?:地区|区域|地市)(?:不限|清空|取消))", text):
            extraction.clear.add("region")
            extraction.touched.add("region")
        if re.search(r"(?:排除|不要|不看).*(?:暴雨|大风|冰雹|高温|寒潮|降雪|沙尘|雾)", text):
            extraction.warnings.append("当前检索协议不支持排除条件，请改用“只看……”明确需要保留的灾种。")
        return extraction

    def _extract_temporal(self, text: str) -> dict[str, Any] | None:
        """解析精确日期、年份、月份范围和季节表达。"""
        values = {"start_date": "", "end_date": "", "years": [], "months": []}
        exact = self._exact_date_range(text)
        if exact:
            values["start_date"], values["end_date"] = exact
            return values

        years: list[int] = []
        year_range = re.search(r"(20\d{2})\s*(?:年)?\s*(?:至|到|—|-|~|～)\s*(20\d{2})\s*年?", text)
        if year_range:
            start_year, end_year = int(year_range.group(1)), int(year_range.group(2))
            if start_year <= end_year and end_year - start_year <= 20:
                years.extend(range(start_year, end_year + 1))
        else:
            years.extend(int(item) for item in re.findall(r"(20\d{2})\s*年", text))
        if "今年" in text:
            years.append(date.today().year)
        if "去年" in text:
            years.append(date.today().year - 1)

        months: list[int] = []
        range_match = re.search(r"(?<!\d)(1[0-2]|0?[1-9])\s*月?\s*(?:至|到|—|-|~|～)\s*(1[0-2]|0?[1-9])\s*月", text)
        if range_match:
            months.extend(self._month_span(int(range_match.group(1)), int(range_match.group(2))))
        else:
            months.extend(int(item) for item in re.findall(r"(?<!\d)(1[0-2]|0?[1-9])\s*月", text))
        for season, season_months in SEASON_MONTHS.items():
            if season in text:
                months.extend(season_months)
        years = list(dict.fromkeys(years))
        months = list(dict.fromkeys(months))
        if not years and not months:
            return None
        values["years"] = years
        values["months"] = months
        return values

    def _exact_date_range(self, text: str) -> tuple[str, str] | None:
        """识别带年份的中文或 ISO 精确日期范围。"""
        iso = re.search(r"(20\d{2}-\d{1,2}-\d{1,2})\s*(?:至|到|—|~|～)\s*(20\d{2}-\d{1,2}-\d{1,2})", text)
        if iso:
            start = self._iso_date(*map(int, iso.group(1).split("-")))
            end = self._iso_date(*map(int, iso.group(2).split("-")))
            return (start, end) if start and end else None
        chinese = re.search(
            r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?\s*(?:至|到|—|~|～)\s*"
            r"(?:(20\d{2})\s*年)?\s*(?:(\d{1,2})\s*月)?\s*(\d{1,2})\s*日",
            text,
        )
        if chinese:
            start_year, start_month, start_day = map(int, chinese.group(1, 2, 3))
            end_year = int(chinese.group(4) or start_year)
            end_month = int(chinese.group(5) or start_month)
            start = self._iso_date(start_year, start_month, start_day)
            end = self._iso_date(end_year, end_month, int(chinese.group(6)))
            return (start, end) if start and end else None
        single = re.search(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
        if single:
            value = self._iso_date(*map(int, single.groups()))
            return (value, value) if value else None
        return None

    def _supplement_with_llm(self, text: str, extraction: NaturalExtraction) -> None:
        """规则没有识别出条件时尝试模型补充，并继续执行严格白名单核验。"""
        if extraction.touched or self.llm_client is None:
            return
        try:
            if not self.llm_client.is_available():
                return
        except Exception:
            return
        question = (
            "从用户消息提取气象个例检索条件，只输出JSON。字段为years、months、start_date、end_date、"
            "disaster_types、cities、areas。不得猜测消息未明确表达的条件；灾种和地区必须逐字来自消息。"
        )
        try:
            raw = str(self.llm_client.answer_with_context(question, [text], max_tokens=500) or "")
            payload = self._json_object(raw)
        except Exception:
            extraction.warnings.append("大模型补充解析不可用，本轮仅采用确定性规则结果。")
            return
        if not payload:
            return
        disasters = [name for name in payload.get("disaster_types", []) if name in DISASTER_NAMES and name in text]
        cities = [name for name in payload.get("cities", []) if name in CITY_NAMES and name in text]
        areas = [name for name in payload.get("areas", []) if name in AREA_NAMES and name in text]
        if disasters:
            extraction.values["disaster_types"] = list(dict.fromkeys(disasters))
            extraction.touched.add("disaster")
        if cities or areas:
            extraction.values["cities"] = list(dict.fromkeys(cities))
            extraction.values["areas"] = list(dict.fromkeys(areas))
            extraction.touched.add("region")
        extraction.parse_status = "llm_supplemented" if extraction.touched else "llm_no_valid_fields"

    def _merge(
        self,
        operation: str,
        extraction: NaturalExtraction,
        committed: StructuredCaseSearchRequest | None,
    ) -> tuple[StructuredCaseSearchRequest, list[str]]:
        """按操作语义合并；触及的维度替换，未触及维度才继承。"""
        if operation == "new" or committed is None:
            resolved = self._request_from_values(extraction.values)
            return resolved, ["建立新的检索条件。"]
        data = committed.model_dump()
        changes: list[str] = []
        if "time" in extraction.touched:
            if "time" in extraction.clear:
                data.update({"start_date": "", "end_date": "", "years": [], "months": []})
                changes.append("时间条件已清空。")
            else:
                incoming = extraction.values
                if incoming.get("start_date") or incoming.get("end_date"):
                    # 完整日期范围优先级最高，不能继续携带快捷年份和月份。
                    data.update(
                        {
                            "start_date": incoming.get("start_date") or "",
                            "end_date": incoming.get("end_date") or "",
                            "years": [],
                            "months": [],
                        }
                    )
                else:
                    # 用户只修改月份时保留已确认年份，只修改年份时保留已确认月份。
                    retained_years = list(data.get("years") or [])
                    retained_months = list(data.get("months") or [])
                    if not retained_years and (data.get("start_date") or data.get("end_date")):
                        retained_years = self._years_from_dates(data.get("start_date"), data.get("end_date"))
                    data.update(
                        {
                            "start_date": "",
                            "end_date": "",
                            "years": list(incoming.get("years") or retained_years),
                            "months": list(incoming.get("months") or retained_months),
                        }
                    )
                changes.append("时间条件已按本轮明确的年份、月份或日期范围更新。")
        if "disaster" in extraction.touched:
            data["disaster_types"] = [] if "disaster" in extraction.clear else list(extraction.values.get("disaster_types") or [])
            changes.append("灾种条件已限定为本轮指定灾种。" if "disaster" not in extraction.clear else "灾种条件已清空。")
        if "region" in extraction.touched:
            data["cities"] = [] if "region" in extraction.clear else list(extraction.values.get("cities") or [])
            data["areas"] = [] if "region" in extraction.clear else list(extraction.values.get("areas") or [])
            changes.append("影响区域已限定为本轮指定范围。" if "region" not in extraction.clear else "影响区域条件已清空。")
        return StructuredCaseSearchRequest.model_validate(data), changes

    def _request_from_values(self, values: dict[str, Any]) -> StructuredCaseSearchRequest:
        """只把检索字段写入现有请求，其余分析参数沿用稳定默认值。"""
        return StructuredCaseSearchRequest(**{field: deepcopy(values.get(field, "" if field in {"start_date", "end_date"} else [])) for field in QUERY_FIELDS})

    def _display_lines(self, request: StructuredCaseSearchRequest) -> list[str]:
        """生成聊天确认卡使用的分行条件。"""
        if request.start_date or request.end_date:
            time_text = f"{request.start_date or '不限'} 至 {request.end_date or '不限'}"
        else:
            parts = []
            if request.years:
                parts.append("、".join(f"{year}年" for year in request.years))
            if request.months:
                parts.append("、".join(f"{month}月" for month in request.months))
            time_text = "；".join(parts) or "不限"
        return [
            f"时间：{time_text}",
            "灾种：" + ("、".join(request.disaster_types) or "不限"),
            "地市：" + ("、".join(request.cities) or "不限"),
            "区域：" + ("、".join(request.areas) or "不限"),
        ]

    def _has_filter(self, request: StructuredCaseSearchRequest) -> bool:
        """判断最终请求是否至少包含一个有效检索维度。"""
        return bool(request.start_date or request.end_date or request.years or request.months or request.disaster_types or request.cities or request.areas)

    def _non_overlapping_terms(self, text: str, terms: tuple[str, ...]) -> list[str]:
        """优先保留长灾种词，避免“雷暴大风”被重复拆成雷暴和大风。"""
        matches: list[tuple[int, int, str]] = []
        occupied: list[tuple[int, int]] = []
        for term in sorted(terms, key=len, reverse=True):
            for match in re.finditer(re.escape(term), text):
                span = match.span()
                if any(span[0] < right and span[1] > left for left, right in occupied):
                    continue
                occupied.append(span)
                matches.append((span[0], span[1], term))
        return list(dict.fromkeys(item[2] for item in sorted(matches)))

    def _month_span(self, start: int, end: int) -> list[int]:
        """展开月份区间，支持11月至次年2月。"""
        if start <= end:
            return list(range(start, end + 1))
        return list(range(start, 13)) + list(range(1, end + 1))

    def _years_from_dates(self, start_date: str, end_date: str) -> list[int]:
        """从已确认精确日期中保留明确年份，供只修改月份时使用。"""
        years = []
        for value in (start_date, end_date):
            match = re.match(r"(20\d{2})-", str(value or ""))
            if match:
                years.append(int(match.group(1)))
        return list(dict.fromkeys(years))

    def _iso_date(self, year: int, month: int, day: int) -> str:
        """校验并格式化日期。"""
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return ""

    def _json_object(self, raw: str) -> dict[str, Any]:
        """从模型返回中提取最终 JSON，不保留思考链。"""
        text = re.sub(r"<think>[\s\S]*?</think>", "", str(raw or ""), flags=re.IGNORECASE).strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


class NaturalConversationStore:
    """将已确认查询和待确认提案持久化到运行时目录。"""

    SESSION_TTL_SECONDS = 7 * 24 * 3600
    PROPOSAL_TTL_SECONDS = 30 * 60
    AGENT_STATE_KEY = "case_multidim"

    def __init__(self, directory: Path | None = None, session_store=None):
        """独立页面使用目录存储，主页面则复用 sessions.sqlite3。"""
        self.session_store = session_store
        self.directory = Path(directory) if directory is not None else None
        if self.session_store is None:
            if self.directory is None:
                raise ValueError("独立多维会话必须配置运行时目录")
            self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def pending_proposal(self, conversation_id: str) -> dict[str, Any] | None:
        """读取仍在有效期内且基于当前版本生成的待确认条件。"""
        with self._lock:
            session = self._load(conversation_id)
            pending = self._valid_pending(session, session.get("pending") if session else None)
            # 统一入口每个请求都会创建新的编排器；如果进程重载或旧状态未写入，
            # 从同一主 session 最近一张确认卡恢复完整提案，避免“确定”退化成新查询。
            if pending is None:
                pending = self._recover_pending_from_messages(conversation_id)
                if pending is not None:
                    session = session or self._new_session(conversation_id)
                    pending["base_version"] = int(session.get("version") or 0)
                    session["pending"] = deepcopy(pending)
                    session["updated_at"] = time()
                    try:
                        self._save(session)
                    except KeyError:
                        return None
            return deepcopy(pending) if pending is not None else None

    def propose(self, conversation_id: str, message: str, parser: NaturalCaseQueryParser) -> NaturalCaseQueryParseResponse:
        """生成待确认提案；该步骤绝不修改已确认查询。"""
        with self._lock:
            session = self._load_or_create(conversation_id)
            pending = self._valid_pending(session, session.get("pending"))
            if pending is None:
                pending = self._recover_pending_from_messages(session["conversation_id"])
                if pending is not None:
                    pending["base_version"] = int(session.get("version") or 0)
                    session["pending"] = deepcopy(pending)
            # 直接调用自然语言解析接口时，“确定/确认”也不能把原提案解析为空并覆盖掉。
            if pending is not None and self._is_confirmation_message(message):
                session["updated_at"] = time()
                self._save(session)
                return self._response_from_pending(session, pending)
            committed_data = session.get("committed_query")
            committed = StructuredCaseSearchRequest.model_validate(committed_data) if committed_data else None
            operation, request, lines, changes, warnings, status, can_confirm = parser.parse(message, committed)
            proposal_id = f"proposal_{uuid4().hex}" if can_confirm else ""
            if can_confirm:
                session["pending"] = {
                    "proposal_id": proposal_id,
                    "base_version": int(session.get("version") or 0),
                    "message": message,
                    "operation": operation,
                    "request": request.model_dump(),
                    "created_at": time(),
                }
            else:
                session["pending"] = None
            session["updated_at"] = time()
            session.setdefault("history", []).append({"role": "user", "content": message, "confirmed": False, "created_at": time()})
            session["history"] = session["history"][-30:]
            self._save(session)
            response_warnings = [
                *warnings,
                *self._response_warnings(message, committed, operation, request, can_confirm),
            ]
            return NaturalCaseQueryParseResponse(
                conversation_id=session["conversation_id"],
                proposal_id=proposal_id,
                operation=operation,
                operation_label=OPERATION_LABELS[operation],
                parsed_request=request,
                display_lines=lines,
                change_summary=changes,
                warnings=list(dict.fromkeys(response_warnings)),
                can_confirm=can_confirm,
                parse_status=status,
            )

    def _response_from_pending(
        self,
        session: dict[str, Any],
        pending: dict[str, Any],
    ) -> NaturalCaseQueryParseResponse:
        """把持久化提案重新构造成确认卡响应，保证跨请求字段完整。"""
        request = StructuredCaseSearchRequest.model_validate(dict(pending.get("request") or {}))
        return NaturalCaseQueryParseResponse(
            conversation_id=session["conversation_id"],
            proposal_id=str(pending.get("proposal_id") or ""),
            operation=str(pending.get("operation") or "new"),
            operation_label=OPERATION_LABELS.get(str(pending.get("operation") or "new"), OPERATION_LABELS["new"]),
            parsed_request=request,
            display_lines=self._display_lines(request),
            change_summary=["已恢复上一张待确认的检索条件。"],
            warnings=[],
            can_confirm=True,
            parse_status="persisted_proposal",
        )

    def _valid_pending(
        self,
        session: dict[str, Any] | None,
        pending: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """统一校验提案有效期和版本，避免旧确认卡覆盖新条件。"""
        if not session or not isinstance(pending, dict) or not pending.get("proposal_id"):
            return None
        try:
            created_at = float(pending.get("created_at") or 0)
            base_version = int(pending.get("base_version") or 0)
            version = int(session.get("version") or 0)
        except (TypeError, ValueError):
            return None
        if time() - created_at > self.PROPOSAL_TTL_SECONDS or base_version != version:
            return None
        if not isinstance(pending.get("request"), dict):
            return None
        return pending

    def _recover_pending_from_messages(self, conversation_id: str) -> dict[str, Any] | None:
        """从主会话最近的确认卡消息恢复提案；终态消息会阻断旧卡复活。"""
        if self.session_store is None or not self._valid_id(conversation_id):
            return None
        try:
            messages = self.session_store.list_messages(conversation_id)
        except Exception:
            return None
        for message in reversed(messages):
            if str(message.get("role") or "") != "assistant":
                continue
            metadata = message.get("metadata") or {}
            if not isinstance(metadata, dict) or metadata.get("agent_type") != "case_multidim_search":
                continue
            status = str(metadata.get("status") or "")
            if status != "awaiting_confirmation":
                # 最新多维终态已经消费或取消提案，不能继续使用更早的确认卡。
                return None
            proposal_id = str(metadata.get("proposal_id") or "")
            request = metadata.get("query_conditions")
            if not proposal_id or not isinstance(request, dict):
                return None
            created_at = self._message_epoch(message.get("created_at"))
            if time() - created_at > self.PROPOSAL_TTL_SECONDS:
                return None
            return {
                "proposal_id": proposal_id,
                "base_version": 0,
                "message": str(metadata.get("proposal_question") or message.get("content") or ""),
                "operation": str(metadata.get("proposal_operation") or "new"),
                "request": deepcopy(request),
                "created_at": created_at,
            }
        return None

    @staticmethod
    def _message_epoch(value: Any) -> float:
        """将 SQLite 消息时间转换为 Unix 秒，解析失败时按当前时间处理。"""
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError, OverflowError):
            return time()

    @staticmethod
    def _is_confirmation_message(message: str) -> bool:
        """识别只用于确认提案的短消息，不把普通业务问题误判为确认。"""
        text = str(message or "").strip()
        # 统一兼容确认卡上的“确认”和“确定”，避免跨请求时重新解析短消息。
        return text in {"确认", "确定", "确认条件", "确定条件", "开始检索", "执行检索", "按此检索", "生成吧"} or text.startswith(("确认", "确定"))

    def confirm(self, conversation_id: str, proposal_id: str, progress_id: str) -> tuple[StructuredCaseSearchRequest, str]:
        """原子确认当前提案，并拒绝过期或基于旧版本生成的确认卡。"""
        with self._lock:
            session = self._load(conversation_id)
            if session is None:
                raise ValueError("聊天会话不存在或已过期，请重新输入检索条件。")
            pending = session.get("pending") or {}
            if pending.get("proposal_id") != proposal_id:
                raise ValueError("该条件确认卡已失效，请重新解析当前消息。")
            if time() - float(pending.get("created_at") or 0) > self.PROPOSAL_TTL_SECONDS:
                raise ValueError("该条件确认卡已过期，请重新解析当前消息。")
            if int(pending.get("base_version") or 0) != int(session.get("version") or 0):
                raise ValueError("会话条件已更新，旧确认卡不能覆盖新的上下文。")
            request_data = dict(pending.get("request") or {})
            request_data["progress_id"] = progress_id
            request = StructuredCaseSearchRequest.model_validate(request_data)
            session["committed_query"] = request.model_copy(update={"progress_id": ""}).model_dump()
            session["version"] = int(session.get("version") or 0) + 1
            session["pending"] = None
            session["updated_at"] = time()
            session.setdefault("history", []).append({
                "role": "assistant",
                "content": "用户已确认结构化检索条件并开始检索。",
                "confirmed": True,
                "query": session["committed_query"],
                "created_at": time(),
            })
            session["history"] = session["history"][-30:]
            self._save(session)
            return request, str(pending.get("message") or "")

    def reset(self, conversation_id: str) -> str:
        """删除旧会话并返回新的会话编号。"""
        with self._lock:
            if self.session_store is not None:
                # 主页面继续使用原 session_id，只清理该会话的多维条件状态。
                self.session_store.delete_agent_state(conversation_id, self.AGENT_STATE_KEY)
                return conversation_id
            path = self._path(conversation_id) if self._valid_id(conversation_id) else None
            if path and path.is_file():
                path.unlink()
            session = self._new_session()
            self._save(session)
            return session["conversation_id"]

    def _response_warnings(
        self,
        message: str,
        committed: StructuredCaseSearchRequest | None,
        operation: str,
        request: StructuredCaseSearchRequest,
        can_confirm: bool,
    ) -> list[str]:
        """生成不依赖模型的会话合并提示。"""
        warnings: list[str] = []
        if committed is not None and operation == "new":
            # 新检索是正常业务行为，不把内部上下文决策作为警告展示给用户。
            pass
        if committed is None and operation != "new":
            warnings.append("当前没有可修改的已确认条件，请先完整描述一次检索。")
        if "北部" in message and not any(term in message for term in ("山西北部", "晋北")):
            warnings.append("请将“北部”明确为“山西北部”或“晋北”。")
        if not can_confirm:
            warnings.append("请补充明确的时间、灾种、地市或区域条件后再检索。")
        if not any(getattr(request, field) for field in QUERY_FIELDS):
            warnings.append("当前最终条件为空。")
        return list(dict.fromkeys(warnings))

    def _load_or_create(self, conversation_id: str) -> dict[str, Any]:
        """读取有效会话，否则建立新会话。"""
        session = self._load(conversation_id) if self._valid_id(conversation_id) else None
        fixed_id = conversation_id if self.session_store is not None and self._valid_id(conversation_id) else ""
        return session or self._new_session(fixed_id)

    def _load(self, conversation_id: str) -> dict[str, Any] | None:
        """读取未过期的会话 JSON。"""
        if not self._valid_id(conversation_id):
            return None
        if self.session_store is not None:
            session = self.session_store.get_agent_state(conversation_id, self.AGENT_STATE_KEY)
            if not session:
                return None
            if time() - float(session.get("updated_at") or 0) > self.SESSION_TTL_SECONDS:
                return None
            return session
        path = self._path(conversation_id)
        if not path.is_file():
            return None
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if time() - float(session.get("updated_at") or 0) > self.SESSION_TTL_SECONDS:
            return None
        return session

    def _new_session(self, conversation_id: str = "") -> dict[str, Any]:
        """建立没有隐式条件的新会话。"""
        now = time()
        return {
            "conversation_id": conversation_id or f"conversation_{uuid4().hex}",
            "version": 0,
            "committed_query": None,
            "pending": None,
            "history": [],
            "created_at": now,
            "updated_at": now,
        }

    def _save(self, session: dict[str, Any]) -> None:
        """原子写入会话，避免进程中断留下半个 JSON。"""
        if self.session_store is not None:
            self.session_store.save_agent_state(
                session["conversation_id"],
                self.AGENT_STATE_KEY,
                session,
            )
            return
        path = self._path(session["conversation_id"])
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _path(self, conversation_id: str) -> Path:
        """把经过校验的会话编号转换为文件路径。"""
        if self.directory is None:
            raise RuntimeError("主页面会话状态不使用文件路径")
        return self.directory / f"{conversation_id}.json"

    def _valid_id(self, conversation_id: str) -> bool:
        """限制会话编号字符，防止路径穿越。"""
        return bool(re.fullmatch(r"[A-Za-z0-9_-]{8,80}", str(conversation_id or "")))
