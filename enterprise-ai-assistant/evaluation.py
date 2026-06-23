"""
evaluation.py
--------------
Critical evaluation layer for the RAG pipeline.

Tracks:
  Retrieval metrics : Recall@5, Precision@5, MRR
  Generation metrics: Faithfulness (LLM-judged), Answer Relevance (embedding
                       similarity), Overall Quality (weighted combination)
  Business metrics  : Success Rate, Hallucination Rate

Anti-hallucination gate: faithfulness < 0.5 => is_hallucination = True,
and callers (main.py) MUST reject the generated answer and return the
standard "no info" response instead.
"""

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langchain.schema import Document

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Thresholds (per spec)
# ----------------------------------------------------------------------
THRESHOLDS = {
    "recall_at_5": 0.85,
    "precision_at_5": 0.80,
    "mrr": 0.75,
    "faithfulness": 0.90,
    "answer_relevance": 0.85,
    "overall_quality": 0.70,
    "success_rate": 0.90,
    "hallucination_rate": 0.05,  # must be BELOW this
}

HALLUCINATION_FAITHFULNESS_CUTOFF = 0.5


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    k: int = 5


@dataclass
class GenerationMetrics:
    faithfulness: float
    answer_relevance: float
    overall_quality: float
    is_hallucination: bool


@dataclass
class QueryEvaluation:
    query: str
    timestamp: str
    retrieval: Optional[RetrievalMetrics] = None
    generation: Optional[GenerationMetrics] = None

    def to_dict(self) -> dict:
        d = {"query": self.query, "timestamp": self.timestamp}
        if self.retrieval:
            d["retrieval"] = asdict(self.retrieval)
        if self.generation:
            d["generation"] = asdict(self.generation)
        return d


class RAGEvaluation:
    """
    Evaluation layer for the hybrid RAG pipeline.

    `llm_generator` must expose `.judge_score(prompt_text) -> float` for
    LLM-judged faithfulness scoring (see llm_generator.LLMGenerator).

    `embeddings` must expose `.embed_query(text) -> List[float]` for
    answer-relevance cosine similarity (any LangChain Embeddings object).
    """

    def __init__(self, llm_generator=None, embeddings=None):
        self.llm_generator = llm_generator
        self.embeddings = embeddings
        self.history: List[QueryEvaluation] = []

    # ------------------------------------------------------------------
    # Retrieval metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_id(doc) -> str:
        """Resolve a stable identifier for a doc, whether it's a Document or a raw string id."""
        if isinstance(doc, Document):
            return doc.metadata.get("chunk_id") or doc.metadata.get("source", "") + str(
                hash(doc.page_content)
            )
        return str(doc)

    def calculate_recall_at_k(
        self, retrieved_docs: List, relevant_docs: List, k: int = 5
    ) -> float:
        """
        Recall@k = |relevant ∩ retrieved[:k]| / |relevant|
        relevant_docs and retrieved_docs may be Document objects or plain
        string ids - both are normalized via _doc_id.
        """
        if not relevant_docs:
            return 0.0
        retrieved_ids = {self._doc_id(d) for d in retrieved_docs[:k]}
        relevant_ids = {self._doc_id(d) for d in relevant_docs}
        hits = retrieved_ids & relevant_ids
        return len(hits) / len(relevant_ids)

    def calculate_precision_at_k(
        self, retrieved_docs: List, relevant_docs: List, k: int = 5
    ) -> float:
        """
        Precision@k = |relevant ∩ retrieved[:k]| / |retrieved[:k]|
        """
        top_k = retrieved_docs[:k]
        if not top_k:
            return 0.0
        retrieved_ids = [self._doc_id(d) for d in top_k]
        relevant_ids = {self._doc_id(d) for d in relevant_docs}
        hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
        return hits / len(top_k)

    def calculate_mrr(self, retrieved_docs: List, relevant_docs: List) -> float:
        """
        Mean Reciprocal Rank (single query): 1 / rank_of_first_relevant_doc.
        0.0 if no relevant doc is found in retrieved_docs.
        """
        relevant_ids = {self._doc_id(d) for d in relevant_docs}
        for rank, doc in enumerate(retrieved_docs, start=1):
            if self._doc_id(doc) in relevant_ids:
                return 1.0 / rank
        return 0.0

    def evaluate_retrieval(
        self, retrieved_docs: List, relevant_docs: List, query: str = "", k: int = 5
    ) -> RetrievalMetrics:
        recall = self.calculate_recall_at_k(retrieved_docs, relevant_docs, k=k)
        precision = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k=k)
        mrr = self.calculate_mrr(retrieved_docs, relevant_docs)
        return RetrievalMetrics(recall_at_k=recall, precision_at_k=precision, mrr=mrr, k=k)

    # ------------------------------------------------------------------
    # Generation metrics
    # ------------------------------------------------------------------

    def calculate_faithfulness(self, query: str, answer: str, context: str) -> float:
        """
        LLM-judged faithfulness: is `answer` fully grounded in `context`?
        Returns a float in [0, 1]. Falls back to a conservative heuristic
        if no llm_generator is configured (e.g. in unit tests).
        """
        if answer.strip() == "I don't have that information in our current documents.":
            # The pipeline itself declined to answer - that's faithful by definition.
            return 1.0

        if self.llm_generator is None:
            return self._heuristic_faithfulness(answer, context)

        from prompt_templates import FAITHFULNESS_JUDGE_PROMPT

        prompt_text = FAITHFULNESS_JUDGE_PROMPT.format(
            context=context, question=query, answer=answer
        )
        try:
            return self.llm_generator.judge_score(prompt_text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Faithfulness judge call failed: %s — falling back to heuristic", exc)
            return self._heuristic_faithfulness(answer, context)

    @staticmethod
    def _heuristic_faithfulness(answer: str, context: str) -> float:
        """
        Lightweight fallback when no LLM judge is available: word-overlap
        ratio between answer and context. Not a substitute for the LLM
        judge, only used so the system degrades gracefully.
        """
        answer_words = set(w.lower() for w in answer.split() if len(w) > 3)
        context_words = set(w.lower() for w in context.split() if len(w) > 3)
        if not answer_words:
            return 0.0
        overlap = answer_words & context_words
        return len(overlap) / len(answer_words)

    def calculate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Embedding-based cosine similarity between query and answer.
        Falls back to a token-overlap heuristic if no embeddings model
        is configured.
        """
        if self.embeddings is None:
            return self._heuristic_relevance(query, answer)

        try:
            query_vec = self.embeddings.embed_query(query)
            answer_vec = self.embeddings.embed_query(answer)
            return self._cosine_similarity(query_vec, answer_vec)
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding-based relevance failed: %s — falling back to heuristic", exc)
            return self._heuristic_relevance(query, answer)

    @staticmethod
    def _heuristic_relevance(query: str, answer: str) -> float:
        query_words = set(w.lower() for w in query.split() if len(w) > 2)
        answer_words = set(w.lower() for w in answer.split() if len(w) > 2)
        if not query_words:
            return 0.0
        overlap = query_words & answer_words
        return len(overlap) / len(query_words)

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        # cosine similarity is in [-1, 1]; clip to [0, 1] for use as a 0-1 quality score
        sim = dot / (norm_a * norm_b)
        return max(0.0, min(1.0, sim))

    def evaluate_generation(
        self,
        query: str,
        answer: str,
        context: str,
        retrieved_docs: Optional[List] = None,
        relevant_docs: Optional[List] = None,
    ) -> GenerationMetrics:
        faithfulness = self.calculate_faithfulness(query, answer, context)
        relevance = self.calculate_answer_relevance(query, answer)

        # Overall quality: weighted combination of faithfulness (most important
        # for anti-hallucination) and answer relevance.
        overall_quality = 0.6 * faithfulness + 0.4 * relevance

        is_hallucination = faithfulness < HALLUCINATION_FAITHFULNESS_CUTOFF

        return GenerationMetrics(
            faithfulness=faithfulness,
            answer_relevance=relevance,
            overall_quality=overall_quality,
            is_hallucination=is_hallucination,
        )

    # ------------------------------------------------------------------
    # End-to-end per-query evaluation (used by main.py /chat/ endpoint)
    # ------------------------------------------------------------------

    def evaluate_query(
        self,
        query: str,
        answer: str,
        retrieved_docs: List,
        relevant_docs: Optional[List] = None,
    ) -> QueryEvaluation:
        """
        Full evaluation of a single live query: retrieval metrics (if
        relevant_docs/ground truth is available) + generation metrics
        (always computed). Logs the result into self.history for the
        dashboard / report.
        """
        context = "\n\n".join(d.page_content for d in retrieved_docs) if retrieved_docs else ""

        retrieval_metrics = None
        if relevant_docs is not None:
            retrieval_metrics = self.evaluate_retrieval(retrieved_docs, relevant_docs, query=query)

        generation_metrics = self.evaluate_generation(
            query=query,
            answer=answer,
            context=context,
            retrieved_docs=retrieved_docs,
            relevant_docs=relevant_docs,
        )

        evaluation = QueryEvaluation(
            query=query,
            timestamp=datetime.now(timezone.utc).isoformat(),
            retrieval=retrieval_metrics,
            generation=generation_metrics,
        )
        self.history.append(evaluation)
        return evaluation

    # ------------------------------------------------------------------
    # Benchmark testing
    # ------------------------------------------------------------------

    def run_benchmark_test(self, pipeline, test_cases: Optional[List[Dict]] = None) -> Dict:
        """
        Run a fixed benchmark dataset of query/expected-answer pairs against
        a live `pipeline` (must expose .answer_query(query) -> (answer, docs)).

        Returns aggregate metrics and a PASS/FAIL verdict per spec:
          PASS if avg_overall_quality > 0.7 AND avg_faithfulness > 0.8
        """
        test_cases = test_cases or self.default_benchmark_dataset()

        results = []
        for case in test_cases:
            query = case["query"]
            try:
                answer, retrieved_docs = pipeline.answer_query(query)
            except Exception as exc:  # noqa: BLE001
                logger.error("Benchmark query failed for %r: %s", query, exc)
                answer, retrieved_docs = "", []

            context = "\n\n".join(d.page_content for d in retrieved_docs) if retrieved_docs else ""
            gen_metrics = self.evaluate_generation(query=query, answer=answer, context=context)

            results.append(
                {
                    "query": query,
                    "expected_answer": case.get("expected_answer", ""),
                    "actual_answer": answer,
                    "faithfulness": gen_metrics.faithfulness,
                    "answer_relevance": gen_metrics.answer_relevance,
                    "overall_quality": gen_metrics.overall_quality,
                    "is_hallucination": gen_metrics.is_hallucination,
                }
            )

        avg_faithfulness = self._safe_avg([r["faithfulness"] for r in results])
        avg_relevance = self._safe_avg([r["answer_relevance"] for r in results])
        avg_overall_quality = self._safe_avg([r["overall_quality"] for r in results])
        hallucination_rate = self._safe_avg(
            [1.0 if r["is_hallucination"] else 0.0 for r in results]
        )

        passed = (
            avg_overall_quality > THRESHOLDS["overall_quality"]
            and avg_faithfulness > 0.8  # explicit benchmark pass bar per spec
        )

        return {
            "status": "PASS" if passed else "FAIL",
            "num_test_cases": len(results),
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_answer_relevance": round(avg_relevance, 4),
            "avg_overall_quality": round(avg_overall_quality, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "results": results,
        }

    @staticmethod
    def default_benchmark_dataset() -> List[Dict]:
        """3-5 known query/expected-answer pairs, per spec."""
        return [
            {
                "query": "What is the enterprise pricing?",
                "expected_answer": "The Enterprise plan costs $499 per month per seat, billed annually.",
            },
            {
                "query": "What ROI can customers expect?",
                "expected_answer": "Customers typically see a 35% reduction in operational costs within six months.",
            },
            {
                "query": "Do you have healthcare solutions?",
                "expected_answer": "Yes, a HIPAA-compliant healthcare module with encrypted PHI storage and audit logging.",
            },
            {
                "query": "What security certifications do you have?",
                "expected_answer": "SOC 2 Type II certified, with annual third-party penetration testing.",
            },
            {
                "query": "What is the refund policy for a 10-year contract?",
                "expected_answer": "I don't have that information in our current documents.",
            },
        ]

    @staticmethod
    def _safe_avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Human-readable evaluation report summarizing self.history."""
        if not self.history:
            return "No queries have been evaluated yet."

        n = len(self.history)
        gen_records = [e.generation for e in self.history if e.generation]
        retrieval_records = [e.retrieval for e in self.history if e.retrieval]

        avg_faithfulness = self._safe_avg([g.faithfulness for g in gen_records])
        avg_relevance = self._safe_avg([g.answer_relevance for g in gen_records])
        avg_quality = self._safe_avg([g.overall_quality for g in gen_records])
        hallucination_rate = self._safe_avg(
            [1.0 if g.is_hallucination else 0.0 for g in gen_records]
        )
        success_rate = 1.0 - hallucination_rate

        lines = [
            "=" * 60,
            "RAG EVALUATION REPORT",
            "=" * 60,
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            f"Total queries evaluated: {n}",
            "",
            "-- Generation Metrics --",
            f"  Avg Faithfulness     : {avg_faithfulness:.3f}  (target > {THRESHOLDS['faithfulness']})"
            f"  {'OK' if avg_faithfulness > THRESHOLDS['faithfulness'] else 'BELOW TARGET'}",
            f"  Avg Answer Relevance : {avg_relevance:.3f}  (target > {THRESHOLDS['answer_relevance']})"
            f"  {'OK' if avg_relevance > THRESHOLDS['answer_relevance'] else 'BELOW TARGET'}",
            f"  Avg Overall Quality  : {avg_quality:.3f}  (target > {THRESHOLDS['overall_quality']})"
            f"  {'OK' if avg_quality > THRESHOLDS['overall_quality'] else 'BELOW TARGET'}",
            "",
            "-- Business Metrics --",
            f"  Success Rate         : {success_rate:.3f}  (target > {THRESHOLDS['success_rate']})"
            f"  {'OK' if success_rate > THRESHOLDS['success_rate'] else 'BELOW TARGET'}",
            f"  Hallucination Rate   : {hallucination_rate:.3f}  (target < {THRESHOLDS['hallucination_rate']})"
            f"  {'OK' if hallucination_rate < THRESHOLDS['hallucination_rate'] else 'ABOVE TARGET - REVIEW PIPELINE'}",
        ]

        if retrieval_records:
            avg_recall = self._safe_avg([r.recall_at_k for r in retrieval_records])
            avg_precision = self._safe_avg([r.precision_at_k for r in retrieval_records])
            avg_mrr = self._safe_avg([r.mrr for r in retrieval_records])
            lines += [
                "",
                "-- Retrieval Metrics --",
                f"  Avg Recall@5         : {avg_recall:.3f}  (target > {THRESHOLDS['recall_at_5']})",
                f"  Avg Precision@5      : {avg_precision:.3f}  (target > {THRESHOLDS['precision_at_5']})",
                f"  Avg MRR              : {avg_mrr:.3f}  (target > {THRESHOLDS['mrr']})",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def save_report(self, filename: str = "evaluation_report.json") -> str:
        """Save the full evaluation history + summary as JSON. Returns the path written."""
        gen_records = [e.generation for e in self.history if e.generation]
        hallucination_rate = self._safe_avg(
            [1.0 if g.is_hallucination else 0.0 for g in gen_records]
        )

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "num_queries": len(self.history),
            "summary": {
                "avg_faithfulness": self._safe_avg([g.faithfulness for g in gen_records]),
                "avg_answer_relevance": self._safe_avg([g.answer_relevance for g in gen_records]),
                "avg_overall_quality": self._safe_avg([g.overall_quality for g in gen_records]),
                "hallucination_rate": hallucination_rate,
                "success_rate": 1.0 - hallucination_rate,
            },
            "thresholds": THRESHOLDS,
            "history": [e.to_dict() for e in self.history],
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Evaluation report saved to %s", filename)
        return filename
