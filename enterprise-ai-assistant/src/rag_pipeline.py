"""
rag_pipeline.py
----------------
Wires together: HybridRetriever -> LLMGenerator -> RAGEvaluation
into a single HybridRAGPipeline used by main.py.

This is where the anti-hallucination guarantee is enforced end-to-end:
every generated answer is faithfulness-checked before being returned,
and rejected (replaced with the "no info" response) if it fails.
"""

import logging
from typing import List, Optional, Tuple

from langchain.schema import Document

from llm_generator import LLMGenerator
from prompt_templates import NO_INFO_RESPONSE
from retriever import HybridRetriever

logger = logging.getLogger(__name__)


class HybridRAGPipeline:
    """
    End-to-end RAG pipeline: hybrid retrieval (BM25 + Vector + RRF) followed
    by grounded generation, with a built-in anti-hallucination gate.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_generator: LLMGenerator,
        evaluator=None,  # RAGEvaluation instance; optional, injected by main.py
        k: int = 5,
    ):
        self.retriever = retriever
        self.llm_generator = llm_generator
        self.evaluator = evaluator
        self.k = k

    def answer_query(
        self, query: str, customer_context: str = "", k: Optional[int] = None
    ) -> Tuple[str, List[Document]]:
        """
        Retrieve relevant documents and generate a grounded answer.
        Returns (answer, retrieved_docs). Does NOT apply the hallucination
        gate itself - see `answer_query_with_evaluation` for the gated,
        evaluation-integrated version used by the API.
        """
        k = k or self.k
        retrieved_docs = self.retriever.retrieve(query, k=k)

        if not retrieved_docs:
            return NO_INFO_RESPONSE, []

        answer = self.llm_generator.generate_answer(
            question=query, documents=retrieved_docs, customer_context=customer_context
        )
        return answer, retrieved_docs

    def answer_query_with_evaluation(
        self, query: str, customer_context: str = "", k: Optional[int] = None
    ) -> dict:
        """
        Full gated pipeline used by the /chat/ endpoint:
          1. retrieve
          2. generate
          3. evaluate (faithfulness, relevance, quality)
          4. if faithfulness < 0.5 -> reject the generated answer and
             replace with the standard "no info" response
        Returns a dict ready to be embedded in the API response body.
        """
        if self.evaluator is None:
            raise RuntimeError(
                "answer_query_with_evaluation requires an evaluator to be configured "
                "on the pipeline (RAGEvaluation instance)."
            )

        answer, retrieved_docs = self.answer_query(query, customer_context=customer_context, k=k)

        evaluation = self.evaluator.evaluate_query(
            query=query, answer=answer, retrieved_docs=retrieved_docs
        )

        final_answer = answer
        if evaluation.generation and evaluation.generation.is_hallucination:
            logger.warning(
                "Hallucination detected for query=%r (faithfulness=%.3f) - rejecting answer.",
                query,
                evaluation.generation.faithfulness,
            )
            final_answer = NO_INFO_RESPONSE

        sources = sorted({d.metadata.get("source", "unknown") for d in retrieved_docs})

        confidence = None
        if evaluation.generation:
            confidence = round(evaluation.generation.overall_quality, 3)

        return {
            "query": query,
            "response": final_answer,
            "sources": sources,
            "confidence": confidence,
            "evaluation": evaluation.to_dict(),
        }

    def generate_email(
        self,
        customer_name: str,
        company: str,
        pain_point: str,
        email_type: str,
        context: str = "",
        k: Optional[int] = None,
    ) -> str:
        """
        Retrieve relevant company info (pricing/features/ROI) related to the
        pain point, then generate a personalized email grounded in it.
        """
        k = k or self.k
        # Use the pain point + email type as the retrieval query so the
        # email pulls in factually relevant company data.
        retrieval_query = f"{pain_point} {email_type}".strip()
        retrieved_docs = self.retriever.retrieve(retrieval_query, k=k) if retrieval_query else []

        return self.llm_generator.generate_email(
            customer_name=customer_name,
            company=company,
            pain_point=pain_point,
            email_type=email_type,
            context_notes=context,
            documents=retrieved_docs,
        )
