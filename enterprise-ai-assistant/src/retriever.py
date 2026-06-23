"""
retriever.py
------------
Hybrid retrieval combining BM25 (keyword) and vector (semantic) search,
fused via LangChain's EnsembleRetriever, which implements Reciprocal Rank
Fusion (RRF) internally.

Default weights: [0.4, 0.6] -> [BM25, Vector], i.e. more semantic weight.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_K = 5


@dataclass
class ScoredDocument:
    """A retrieved document with its fused rank position (RRF-based)."""
    document: Document
    rank: int
    rrf_score: float = 0.0


class HybridRetriever:
    """
    Combines BM25Retriever (keyword) and a Chroma-backed vector retriever
    (semantic) into a single EnsembleRetriever using Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever(documents, vectorstore)
        docs = retriever.retrieve("enterprise pricing", k=5)
        scored = retriever.retrieve_with_scores("enterprise pricing", k=5)
    """

    def __init__(
        self,
        documents: List[Document],
        vectorstore: Chroma,
        weights: Tuple[float, float] = (DEFAULT_BM25_WEIGHT, DEFAULT_VECTOR_WEIGHT),
        k: int = DEFAULT_K,
    ):
        if not documents:
            raise ValueError("HybridRetriever requires a non-empty list of documents for BM25.")
        if abs(sum(weights) - 1.0) > 1e-6:
            logger.warning("Retriever weights %s do not sum to 1.0", weights)

        self.k = k
        self.weights = weights

        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = k

        self.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=list(weights),
        )

        logger.info(
            "HybridRetriever initialized: k=%d, weights(bm25,vector)=%s, corpus_size=%d",
            k,
            weights,
            len(documents),
        )

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Return top-k fused documents for `query` using hybrid BM25+vector
        search combined via RRF (EnsembleRetriever).
        """
        k = k or self.k
        results = self.ensemble_retriever.invoke(query)
        return results[:k]

    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[ScoredDocument]:
        """
        Same as retrieve(), but also computes an explicit RRF score per
        document so callers (e.g. the evaluation layer) can inspect
        ranking quality. EnsembleRetriever fuses internally; here we
        recompute RRF scores transparently for observability.
        """
        k = k or self.k
        rrf_const = 60  # standard RRF constant

        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)

        scores: dict[str, float] = {}
        doc_lookup: dict[str, Document] = {}

        for weight, doc_list in zip(self.weights, [bm25_docs, vector_docs]):
            for rank, doc in enumerate(doc_list):
                key = self._doc_key(doc)
                doc_lookup[key] = doc
                scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rrf_const + rank + 1))

        ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)[:k]

        return [
            ScoredDocument(document=doc_lookup[key], rank=i, rrf_score=scores[key])
            for i, key in enumerate(ranked_keys)
        ]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """Stable identity key for dedup across BM25/vector result sets."""
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id:
            return chunk_id
        return f"{doc.metadata.get('source', 'unknown')}::{hash(doc.page_content)}"
