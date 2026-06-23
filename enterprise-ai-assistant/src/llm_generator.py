"""
llm_generator.py
-----------------
Initializes the LLM backend (OpenAI or Groq) and provides generation
methods for query answering and email generation, using the prompt
templates in prompt_templates.py.
"""

import logging
import os
from typing import List, Optional

from langchain.schema import Document

from prompt_templates import (
    EMAIL_GENERATION_PROMPT,
    NO_INFO_RESPONSE,
    QUERY_ANSWER_PROMPT,
)

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.3  # low temperature for factual query answering
EMAIL_TEMPERATURE = 0.6  # slightly higher for natural-sounding emails
DEFAULT_MAX_TOKENS = 600


class LLMGenerator:
    """
    Thin wrapper around a chat LLM (OpenAI GPT or Groq) used for:
      - grounded query answering
      - personalized email generation

    llm_provider:
        "openai" (default) -> requires OPENAI_API_KEY
        "groq"              -> requires GROQ_API_KEY (free tier available)
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.llm_provider = (llm_provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.llm = self._init_llm()

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    def _init_llm(self):
        if self.llm_provider == "openai":
            from langchain_openai import ChatOpenAI

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Set it in .env or use llm_provider='groq'."
                )
            return ChatOpenAI(
                model=self.model_name or "gpt-4o-mini",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=api_key,
            )

        if self.llm_provider == "groq":
            try:
                from langchain_groq import ChatGroq
            except ImportError as exc:
                raise RuntimeError(
                    "langchain-groq is not installed. Run: pip install langchain-groq"
                ) from exc

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set. Set it in .env.")
            return ChatGroq(
                model=self.model_name or "llama-3.1-8b-instant",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                groq_api_key=api_key,
            )

        raise ValueError(f"Unknown llm_provider '{self.llm_provider}'. Use 'openai' or 'groq'.")

    # ------------------------------------------------------------------
    # Generation methods
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        documents: List[Document],
        customer_context: str = "",
    ) -> str:
        """
        Generate a grounded answer to `question` using only `documents` as context.
        Returns the standard "no info" response if no documents were retrieved.
        """
        if not documents:
            return NO_INFO_RESPONSE

        context = self._format_context(documents)
        prompt_text = QUERY_ANSWER_PROMPT.format(
            context=context,
            customer_context=customer_context or "N/A",
            question=question,
        )
        response = self.llm.invoke(prompt_text)
        answer = self._extract_text(response).strip()
        return answer

    def generate_email(
        self,
        customer_name: str,
        company: str,
        pain_point: str,
        email_type: str,
        context_notes: str,
        documents: Optional[List[Document]] = None,
    ) -> str:
        """
        Generate a personalized 200-300 word sales email grounded in
        retrieved company documents (pricing/features/ROI), addressing
        the customer's specific pain point.
        """
        context = self._format_context(documents) if documents else "No additional company data retrieved."

        prompt_text = EMAIL_GENERATION_PROMPT.format(
            email_type=email_type,
            context=context,
            customer_name=customer_name,
            company=company,
            pain_point=pain_point,
            context_notes=context_notes or "N/A",
        )

        # Use a slightly higher temperature for natural tone in emails
        original_temp = getattr(self.llm, "temperature", None)
        try:
            if hasattr(self.llm, "temperature"):
                self.llm.temperature = EMAIL_TEMPERATURE
            response = self.llm.invoke(prompt_text)
        finally:
            if original_temp is not None and hasattr(self.llm, "temperature"):
                self.llm.temperature = original_temp

        return self._extract_text(response).strip()

    def judge_score(self, prompt_text: str) -> float:
        """
        Generic helper for LLM-as-judge style numeric scoring (used by
        evaluation.py's faithfulness check). Returns a float in [0, 1],
        defaulting to 0.0 if parsing fails (fail-safe / conservative).
        """
        response = self.llm.invoke(prompt_text)
        text = self._extract_text(response).strip()
        try:
            # Extract the first float-looking token in case the model adds extra text
            import re

            match = re.search(r"[01](?:\.\d+)?|0?\.\d+", text)
            if match:
                score = float(match.group(0))
                return max(0.0, min(1.0, score))
        except (ValueError, AttributeError):
            pass
        logger.warning("Could not parse judge score from LLM output: %r — defaulting to 0.0", text)
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _extract_text(response) -> str:
        # ChatOpenAI / ChatGroq both return an AIMessage with .content
        return getattr(response, "content", str(response))
