"""
document_loader.py
-------------------
Loads documents from disk (PDF, TXT, CSV) and from individual uploaded files,
returning clean LangChain Document objects ready for chunking.
"""

import logging
import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""


class DocumentLoader:
    """
    Loads documents of multiple formats (PDF, TXT, CSV) from a directory
    or a single file path, and returns a clean list of LangChain Documents.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_directory(self) -> List[Document]:
        """
        Load every supported file in self.data_dir.
        Skips unsupported file types with a warning instead of failing hard.
        """
        if not self.data_dir.exists():
            raise DocumentLoadError(f"Data directory does not exist: {self.data_dir}")

        all_docs: List[Document] = []
        files = sorted(p for p in self.data_dir.iterdir() if p.is_file())

        if not files:
            logger.warning("No files found in %s", self.data_dir)
            return all_docs

        for file_path in files:
            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                logger.warning("Skipping unsupported file type: %s", file_path.name)
                continue
            try:
                docs = self.load_file(str(file_path))
                all_docs.extend(docs)
                logger.info("Loaded %d page/row chunks from %s", len(docs), file_path.name)
            except Exception as exc:  
                logger.error("Failed to load %s: %s", file_path.name, exc)

        return all_docs

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file (PDF, TXT, or CSV) and return clean Documents
        with normalized metadata (source filename).
        """
        path = Path(file_path)
        if not path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise DocumentLoadError(
                f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        try:
            if ext == ".pdf":
                docs = PyPDFLoader(str(path)).load()
            elif ext == ".txt":
                docs = TextLoader(str(path), encoding="utf-8").load()
            elif ext == ".csv":
                docs = CSVLoader(str(path)).load()
            else:  
                raise DocumentLoadError(f"Unhandled extension: {ext}")
        except Exception as exc:
            raise DocumentLoadError(f"Error parsing {path.name}: {exc}") from exc

        return self._clean_documents(docs, source_name=path.name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_documents(docs: List[Document], source_name: str) -> List[Document]:
        """Normalize whitespace and ensure consistent 'source' metadata."""
        cleaned: List[Document] = []
        for doc in docs:
            text = " ".join(doc.page_content.split())  # collapse whitespace
            if not text.strip():
                continue
            metadata = dict(doc.metadata)
            metadata["source"] = source_name
            cleaned.append(Document(page_content=text, metadata=metadata))
        return cleaned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DocumentLoader(data_dir=os.environ.get("DATA_DIR", "data"))
    documents = loader.load_directory()
    print(f"Loaded {len(documents)} documents total.")
    for d in documents[:3]:
        print("---")
        print(d.metadata)
        print(d.page_content[:200])
