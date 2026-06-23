"""
embedding_manager.py
---------------------
Generates embeddings (OpenAI by default, HuggingFace as a free fallback)
and persists them in a ChromaDB collection for vector retrieval.
"""

import logging
import os
from typing import List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "enterprise_docs_hybrid"
DEFAULT_PERSIST_DIR = "chroma_db"


class EmbeddingManager:
    """
    Wraps embedding generation + ChromaDB persistence.

    embedding_provider:
        "openai"      -> requires OPENAI_API_KEY, uses text-embedding-3-small
        "huggingface" -> free, local, uses sentence-transformers/all-MiniLM-L6-v2
                          (requires the `sentence-transformers` package)
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        embedding_provider: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_provider = (
            embedding_provider or os.environ.get("EMBEDDING_PROVIDER", "openai")
        ).lower()

        self.embeddings = self._init_embeddings()
        self._vectorstore: Optional[Chroma] = None

    # ------------------------------------------------------------------
    # Embedding backend selection
    # ------------------------------------------------------------------

    def _init_embeddings(self):
        if self.embedding_provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Set it in .env or pass "
                    "embedding_provider='huggingface' to use a free local model."
                )
            return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

        if self.embedding_provider == "huggingface":
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface embeddings require `sentence-transformers` to be installed: "
                    "pip install sentence-transformers"
                ) from exc
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        raise ValueError(
            f"Unknown embedding_provider '{self.embedding_provider}'. "
            "Use 'openai' or 'huggingface'."
        )

    # ------------------------------------------------------------------
    # Vector store lifecycle
    # ------------------------------------------------------------------

    def build_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Embed `documents` and persist them into a fresh/updated Chroma collection.
        """
        if not documents:
            raise ValueError("Cannot build vectorstore from an empty document list.")

        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        self._vectorstore.persist() 
        logger.info(
            "Persisted %d chunks to Chroma collection '%s' at '%s'",
            len(documents),
            self.collection_name,
            self.persist_directory,
        )
        return self._vectorstore

    def load_vectorstore(self) -> Chroma:
        """Load an existing persisted Chroma collection without re-embedding."""
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        return self._vectorstore

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents (e.g. from an upload) to an existing collection."""
        if self._vectorstore is None:
            self._vectorstore = self.load_vectorstore()
        self._vectorstore.add_documents(documents)
        self._vectorstore.persist()
        logger.info("Added %d new chunks to collection '%s'", len(documents), self.collection_name)

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = self.load_vectorstore()
        return self._vectorstore

    def as_retriever(self, k: int = 5):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
