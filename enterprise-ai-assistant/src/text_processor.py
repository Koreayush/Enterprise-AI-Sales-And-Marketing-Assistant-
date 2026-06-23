"""
text_processor.py
------------------
Splits documents into retrieval-sized chunks and cleans text content.
Chunk size 500 chars / overlap 50 chars, per spec.
"""

import logging
import re
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


class TextProcessor:
    """
    Wraps RecursiveCharacterTextSplitter with project-specific defaults
    and adds light text cleaning (whitespace, stray control characters).
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def clean_text(self, text: str) -> str:
        """Remove excess whitespace and non-printable/control characters."""
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
        text = re.sub(r"\s+", " ", text)  # collapse whitespace
        return text.strip()

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean and split a list of Documents into chunks.
        Each output chunk retains the source metadata of its parent,
        plus a chunk_index for traceability.
        """
        cleaned_docs = []
        for doc in documents:
            cleaned_text = self.clean_text(doc.page_content)
            if not cleaned_text:
                continue
            cleaned_docs.append(Document(page_content=cleaned_text, metadata=dict(doc.metadata)))

        chunks = self.splitter.split_documents(cleaned_docs)

        # Add stable chunk_id / chunk_index metadata, useful for evaluation & citations
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            source = chunk.metadata.get("source", "unknown")
            chunk.metadata.setdefault("chunk_id", f"{source}::chunk_{i}")

        logger.info(
            "Split %d documents into %d chunks (size=%d, overlap=%d)",
            len(documents),
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = [
        Document(
            page_content="  This   is\n\na   sample document. " * 20,
            metadata={"source": "sample.txt"},
        )
    ]
    processor = TextProcessor()
    out_chunks = processor.split_documents(sample)
    print(f"Produced {len(out_chunks)} chunks")
    print(out_chunks[0].page_content[:120])
    print(out_chunks[0].metadata)
