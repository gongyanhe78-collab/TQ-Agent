from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Any

from backend.app.models import DocumentChunk


class DocumentChunkIndexer:
    """把 PDF 文档化、切分成 chunk，并调用嵌入模型生成向量。"""

    def __init__(
        self,
        embedding_client,
        document_loader: Callable[[Path], Any] | None = None,
        text_splitter: Callable[[Any], Iterable[Any]] | None = None,
    ):
        """注入嵌入客户端，并允许测试替换 PDF 文档化和切分逻辑。"""
        self.embedding_client = embedding_client
        self.document_loader = document_loader or self._load_pdf_documents
        self.text_splitter = text_splitter or self._split_documents

    def index_pdf(self, pdf_path: Path) -> list[DocumentChunk]:
        """把单个 PDF 转换为带向量的 DocumentChunk 列表。"""
        pdf_path = Path(pdf_path)
        docs = self.document_loader(pdf_path)
        raw_chunks = self.text_splitter(docs)
        texts = [self._chunk_to_text(chunk).strip() for chunk in raw_chunks]
        texts = [text for text in texts if text]
        embeddings = self.embedding_client.embed_documents(texts) if texts else []
        chunks: list[DocumentChunk] = []
        for index, (text, embedding) in enumerate(zip(texts, embeddings), start=1):
            chunks.append(
                DocumentChunk(
                    source_pdf=pdf_path.name,
                    chunk_id=f"{pdf_path.stem}-chunk-{index:03d}",
                    chunk_no=index,
                    content=text,
                    file_path=str(pdf_path),
                    embedding=embedding,
                )
            )
        return chunks

    def index_pdfs(self, pdf_paths: list[Path]) -> list[DocumentChunk]:
        """批量处理多个 PDF，并合并返回所有文档 chunk。"""
        chunks: list[DocumentChunk] = []
        for pdf_path in pdf_paths:
            chunks.extend(self.index_pdf(pdf_path))
        return chunks

    def _load_pdf_documents(self, pdf_path: Path):
        """优先使用 Unstructured 文档化，失败或为空时回退到 pypdf 页文本。"""
        try:
            docs = self._load_pdf_with_unstructured(pdf_path)
            if docs:
                return docs
        except Exception:
            pass
        return self._load_pdf_with_pypdf(pdf_path)

    def _load_pdf_with_unstructured(self, pdf_path: Path):
        """使用 Unstructured 将 PDF 转换为 LangChain Document 列表。"""
        try:
            from langchain_core.documents import Document
            from unstructured.partition.pdf import partition_pdf
        except ImportError as exc:
            raise RuntimeError(
                "unstructured and langchain_core are required for document PDF indexing."
            ) from exc

        elements = partition_pdf(filename=str(pdf_path))
        docs = []
        for element in elements:
            text = str(element).strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source_pdf": pdf_path.name}))
        return docs

    def _load_pdf_with_pypdf(self, pdf_path: Path):
        """使用 pypdf 将可抽取文本的 PDF 转换为页级 Document。"""
        try:
            from langchain_core.documents import Document
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf and langchain_core are required for PDF text loading.") from exc

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
        return docs

    def _split_documents(self, docs):
        """使用 RecursiveCharacterTextSplitter 按中文友好的分隔符切分文档。"""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise RuntimeError("langchain_text_splitters is required for document chunking.") from exc

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
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

    def _chunk_to_text(self, chunk) -> str:
        """兼容 LangChain Document、字符串和其他对象，统一取出 chunk 正文。"""
        if isinstance(chunk, str):
            return chunk
        page_content = getattr(chunk, "page_content", None)
        if page_content is not None:
            return str(page_content)
        return str(chunk)
