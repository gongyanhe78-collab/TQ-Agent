"""把命中个例关联的原文片段和图片元数据装配成报告证据。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.app.models import DocumentChunk, StandardCase


@dataclass
class CaseEvidenceBundle:
    """单个标准化个例可用于报告生成的证据摘要。"""

    case_id: str
    title: str
    text_evidence: list[str] = field(default_factory=list)
    image_evidence: list[str] = field(default_factory=list)

    def public_text(self, limit: int = 2) -> str:
        """返回不暴露底层编号的业务化证据描述。"""
        parts = []
        parts.extend(self.text_evidence[:limit])
        parts.extend(self.image_evidence[:limit])
        return "；".join(part for part in parts if part)


class CaseEvidenceBuilder:
    """按 source_chunk_ids 和 evidence_image_ids 精确回填报告证据。"""

    def __init__(self, image_store):
        """保存图片元数据存储依赖。"""
        self.image_store = image_store

    def build(
        self,
        cases: list[StandardCase],
        chunks_by_id: dict[str, DocumentChunk],
    ) -> dict[str, CaseEvidenceBundle]:
        """为每个命中个例生成不暴露底层存储编号的证据包。"""
        return {case.case_id: self._case_bundle(case, chunks_by_id) for case in cases}

    def _case_bundle(
        self,
        case: StandardCase,
        chunks_by_id: dict[str, DocumentChunk],
    ) -> CaseEvidenceBundle:
        """装配单个个例的原文证据和图片证据。"""
        text_evidence = [
            self._text_excerpt(chunks_by_id[chunk_id].content)
            for chunk_id in case.source_chunk_ids
            if chunk_id in chunks_by_id and chunks_by_id[chunk_id].content
        ]
        image_evidence = [
            self._image_description(record)
            for record in self.image_store.list_by_image_ids(case.evidence_image_ids)
        ]
        return CaseEvidenceBundle(
            case_id=case.case_id,
            title=case.title,
            text_evidence=self._dedupe(text_evidence)[:3],
            image_evidence=self._dedupe(image_evidence)[:3],
        )

    def _text_excerpt(self, text: str) -> str:
        """从原文片段中抽取适合报告阅读的短证据句。"""
        cleaned = self._clean_text(text)
        sentences = [item.strip() for item in re.split(r"[。！？\n]", cleaned) if item.strip()]
        if not sentences:
            return ""
        focused = self._choose_sentence(sentences)
        return f"{focused[:120]}。"

    def _choose_sentence(self, sentences: list[str]) -> str:
        """优先选择包含业务关键词的证据句。"""
        keywords = ("暴雨", "大风", "雷暴", "冰雹", "强对流", "降水", "回波", "风速", "影响", "预警")
        for sentence in sentences:
            if any(keyword in sentence for keyword in keywords):
                return sentence
        return sentences[0]

    def _image_description(self, record) -> str:
        """把图片元数据转成不含 image_id 的业务证据描述。"""
        caption = self._clean_text(str(getattr(record, "caption", "") or ""))
        nearby_text = self._clean_text(str(getattr(record, "nearby_text", "") or ""))
        if caption:
            return f"配套图件包括{caption[:100]}。"
        if nearby_text:
            return f"配套图件说明涉及{nearby_text[:100]}。"
        return ""

    def _clean_text(self, text: str) -> str:
        """清理换行、连续空白和常见底层编号。"""
        value = re.sub(r"\s+", " ", text or "").strip()
        value = re.sub(r"\b[\w\u4e00-\u9fff-]+-chunk-\d+\b", "", value)
        value = re.sub(r"\b[\w\u4e00-\u9fff-]+-image-\d+\b", "", value)
        return value.strip(" ，,。；;")

    def _dedupe(self, values: list[str]) -> list[str]:
        """保持顺序去重并去掉空值。"""
        results = []
        seen = set()
        for value in values:
            if value and value not in seen:
                seen.add(value)
                results.append(value)
        return results
