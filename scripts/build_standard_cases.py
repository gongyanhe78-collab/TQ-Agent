from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.services.document_store import ChromaDocumentChunkStore
from backend.app.services.image_extraction import ImageEvidenceStore
from backend.app.services.llm_client import DashScopeChatClient
from backend.app.services.standard_case_builder import StandardCaseBuilder
from backend.app.services.standard_case_store import JsonStandardCaseStore


def build_standard_cases(use_llm: bool = True) -> dict:
    """从 document_index/chunks.json 构建 data/standard_cases.json。"""
    document_store = ChromaDocumentChunkStore(
        persist_dir=settings.document_index_dir,
        collection_name=settings.document_collection_name,
    )
    image_store = ImageEvidenceStore(settings.image_metadata_path)
    llm_client = DashScopeChatClient()
    builder = StandardCaseBuilder(
        image_store=image_store,
        llm_client=llm_client,
        use_llm=use_llm,
    )
    chunks = document_store.list_chunks()
    cases = builder.build(chunks)
    store = JsonStandardCaseStore(settings.standard_cases_path)
    store.replace_cases(cases)
    return {
        "source_chunk_count": len(chunks),
        "case_count": len(cases),
        "case_ids": [case.case_id for case in cases],
        "storage_path": str(settings.standard_cases_path),
        "llm_requested": use_llm,
        "llm_available": llm_client.is_available(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standardized weather case records.")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Use rule-based extraction only; do not call the configured chat model.",
    )
    args = parser.parse_args()
    result = build_standard_cases(use_llm=not args.no_llm)
    print("Standard case build complete.")
    for key, value in result.items():
        if key == "case_ids":
            print(f"{key}: {len(value)} ids")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
