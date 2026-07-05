from __future__ import annotations

import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import settings
from backend.app.services.document_indexing import DocumentChunkIndexer
from backend.app.services.document_store import ChromaDocumentChunkStore
from backend.app.services.embedding_client import DashScopeEmbeddingClient


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def clear_document_index() -> None:
    """清空配置中的 document_index 目录，保留目录本身。"""
    target = settings.document_index_dir.resolve()
    expected = (settings.data_dir / "document_index").resolve()
    if target != expected:
        raise RuntimeError(f"Unexpected document_index path: {target}")

    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def split_documents_500_50(docs):
    """按 500 字符切分，并允许相邻 chunk 重复 50 字符。"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError("langchain_text_splitters is required for document chunking.") from exc

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
            "，",
            " ",
            "",
        ],
        keep_separator=True,
    )
    return text_splitter.split_documents(docs)


def load_pdf_with_pypdf(pdf_path: Path):
    """将 PDF 每页文本转换为 LangChain Document，适配当前文本型 PDF。"""
    try:
        from langchain_core.documents import Document
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf and langchain_core are required for PDF document loading.") from exc

    reader = PdfReader(str(pdf_path))
    docs = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source_pdf": pdf_path.name, "page": page_index},
                )
            )
    if not docs:
        raise RuntimeError(f"No extractable text found in {pdf_path}")
    return docs


def find_resource_pdfs() -> list[Path]:
    pdfs = sorted(settings.resource_dir.rglob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDF files found in {settings.resource_dir}")
    return pdfs


def build_document_index() -> dict:
    clear_document_index()

    embedding_client = DashScopeEmbeddingClient(model=settings.embedding_model)
    if not embedding_client.is_available():
        raise RuntimeError("Missing DashScope API key; refusing to build fallback hash embeddings.")

    indexer = DocumentChunkIndexer(
        embedding_client=embedding_client,
        document_loader=load_pdf_with_pypdf,
        text_splitter=split_documents_500_50,
    )
    pdf_paths = find_resource_pdfs()

    chunks = []
    for pdf_path in pdf_paths:
        print(f"Indexing {pdf_path.name} ...", flush=True)
        pdf_chunks = indexer.index_pdf(pdf_path)
        print(f"  chunks: {len(pdf_chunks)}", flush=True)
        chunks.extend(pdf_chunks)

    store = ChromaDocumentChunkStore(
        persist_dir=settings.document_index_dir,
        collection_name=settings.document_collection_name,
    )
    store.replace_chunks(chunks)
    info = store.collection_info()
    info.update(
        {
            "pdf_count": len(pdf_paths),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": settings.embedding_model,
            "persist_dir": str(settings.document_index_dir),
            "collection_name": settings.document_collection_name,
        }
    )
    return info


def main() -> None:
    info = build_document_index()
    print("Document index build complete.")
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
