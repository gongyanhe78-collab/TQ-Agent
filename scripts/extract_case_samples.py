from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.models import ExtractedCase
from backend.app.services.case_splitter import CaseSplitter
from backend.app.services.pdf_reader import PdfReaderService


@dataclass
class ExtractionReportItem:
    source_pdf: str
    title: str
    sample_id: str
    sample_type: str


def write_case_sample(case: ExtractedCase, output_dir: Path) -> Path:
    output_path = output_dir / f"{case.case_id}.txt"
    payload = "\n".join(
        [
            "sample_type: case",
            f"source_pdf: {case.source_pdf}",
            f"case_id: {case.case_id}",
            f"title: {case.title}",
            f"date_range: {case.date_range}",
            "",
            case.content.strip(),
            "",
        ]
    )
    output_path.write_text(payload, encoding="utf-8")
    return output_path


def write_qa_supplement_sample(
    source_pdf: str,
    sample_index: int,
    question: str,
    answer: str,
    output_dir: Path,
) -> Path:
    source_stem = Path(source_pdf).stem
    sample_id = f"{source_stem}-qa-summary-{sample_index:02d}"
    output_path = output_dir / f"{sample_id}.txt"
    payload = "\n".join(
        [
            "sample_type: qa_supplement",
            f"source_pdf: {source_pdf}",
            f"case_id: {sample_id}",
            "title: 综合问答样例",
            "date_range: ",
            "",
            f"question: {question.strip()}",
            "",
            f"answer: {answer.strip()}",
            "",
        ]
    )
    output_path.write_text(payload, encoding="utf-8")
    return output_path


def extract_pdf_cases(resource_dir: Path, output_dir: Path) -> list[ExtractionReportItem]:
    reader = PdfReaderService()
    splitter = CaseSplitter()
    report: list[ExtractionReportItem] = []
    for pdf_path in sorted(resource_dir.glob("*.pdf")):
        text = reader.read_text(pdf_path)
        cases = splitter.split_cases(text, pdf_path.name)
        for case in cases:
            write_case_sample(case, output_dir)
            report.append(
                ExtractionReportItem(
                    source_pdf=case.source_pdf,
                    title=case.title,
                    sample_id=case.case_id,
                    sample_type="case",
                )
            )
    return report


def add_qa_supplements(
    qa_path: Path,
    case_report: list[ExtractionReportItem],
    output_dir: Path,
) -> list[ExtractionReportItem]:
    if not qa_path.exists():
        return []
    qa_groups = json.loads(qa_path.read_text(encoding="utf-8"))
    case_counts_by_pdf: dict[str, int] = {}
    for item in case_report:
        case_counts_by_pdf[item.source_pdf] = case_counts_by_pdf.get(item.source_pdf, 0) + 1

    supplements: list[ExtractionReportItem] = []
    for group in qa_groups:
        source_pdf = group.get("pdf_filename", "")
        samples = group.get("samples", [])
        extra_count = len(samples) - case_counts_by_pdf.get(source_pdf, 0)
        if extra_count <= 0:
            continue
        for offset, sample in enumerate(samples[-extra_count:], start=1):
            output_path = write_qa_supplement_sample(
                source_pdf=source_pdf,
                sample_index=offset,
                question=sample.get("question", ""),
                answer=sample.get("answer", ""),
                output_dir=output_dir,
            )
            supplements.append(
                ExtractionReportItem(
                    source_pdf=source_pdf,
                    title="综合问答样例",
                    sample_id=output_path.stem,
                    sample_type="qa_supplement",
                )
            )
    return supplements


def extract_samples(
    resource_dir: Path,
    output_dir: Path,
    qa_path: Path | None = None,
    include_qa_supplements: bool = True,
    clean: bool = True,
) -> list[ExtractionReportItem]:
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = extract_pdf_cases(resource_dir, output_dir)
    if include_qa_supplements and qa_path is not None:
        report.extend(add_qa_supplements(qa_path, report, output_dir))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF weather case samples into TXT files.")
    parser.add_argument("--resource-dir", type=Path, default=settings.resource_dir)
    parser.add_argument("--output-dir", type=Path, default=settings.data_dir / "extracted_samples")
    parser.add_argument("--qa-path", type=Path, default=settings.project_root / "weather_qa_results.json")
    parser.add_argument("--no-qa-supplement", action="store_true")
    parser.add_argument("--no-clean", action="store_true")
    args = parser.parse_args()

    report = extract_samples(
        resource_dir=args.resource_dir,
        output_dir=args.output_dir,
        qa_path=args.qa_path,
        include_qa_supplements=not args.no_qa_supplement,
        clean=not args.no_clean,
    )

    counts: dict[str, int] = {}
    for item in report:
        counts[item.sample_type] = counts.get(item.sample_type, 0) + 1
        print(f"[{item.sample_type}] {item.sample_id} <- {item.source_pdf} :: {item.title}")
    print(f"total={len(report)} " + " ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
