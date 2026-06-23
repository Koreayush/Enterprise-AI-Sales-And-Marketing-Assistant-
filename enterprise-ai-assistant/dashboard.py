"""
dashboard.py
------------
Generates a Plotly bar-chart dashboard comparing actual evaluation metrics
against their target thresholds, saved as evaluation_dashboard.html.
"""

import logging
from typing import Optional

import plotly.graph_objects as go

from evaluation import THRESHOLDS, RAGEvaluation

logger = logging.getLogger(__name__)

METRIC_LABELS = {
    "avg_faithfulness": "Faithfulness",
    "avg_answer_relevance": "Answer Relevance",
    "avg_overall_quality": "Overall Quality",
    "success_rate": "Success Rate",
}

THRESHOLD_KEY_MAP = {
    "avg_faithfulness": "faithfulness",
    "avg_answer_relevance": "answer_relevance",
    "avg_overall_quality": "overall_quality",
    "success_rate": "success_rate",
}


def build_dashboard(evaluator: RAGEvaluation, output_path: str = "evaluation_dashboard.html") -> str:
    """
    Build a Plotly bar chart comparing actual vs. target metrics from the
    evaluator's history, and save it as a standalone HTML file.
    """
    gen_records = [e.generation for e in evaluator.history if e.generation]
    retrieval_records = [e.retrieval for e in evaluator.history if e.retrieval]

    if not gen_records:
        logger.warning("No evaluation history available - generating an empty placeholder dashboard.")
        fig = go.Figure()
        fig.update_layout(title="RAG Evaluation Dashboard — No data yet")
        fig.write_html(output_path)
        return output_path

    hallucination_rate = sum(1.0 for g in gen_records if g.is_hallucination) / len(gen_records)
    success_rate = 1.0 - hallucination_rate

    summary = {
        "avg_faithfulness": sum(g.faithfulness for g in gen_records) / len(gen_records),
        "avg_answer_relevance": sum(g.answer_relevance for g in gen_records) / len(gen_records),
        "avg_overall_quality": sum(g.overall_quality for g in gen_records) / len(gen_records),
        "success_rate": success_rate,
    }

    metric_names = [METRIC_LABELS[k] for k in summary]
    actual_values = [round(v, 4) for v in summary.values()]
    target_values = [THRESHOLDS[THRESHOLD_KEY_MAP[k]] for k in summary]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Actual",
            x=metric_names,
            y=actual_values,
            marker_color="#2563eb",
            text=[f"{v:.2f}" for v in actual_values],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Target",
            x=metric_names,
            y=target_values,
            marker_color="#94a3b8",
            text=[f"{v:.2f}" for v in target_values],
            textposition="outside",
        )
    )

    # Hallucination rate gets its own callout since lower is better (inverted target)
    fig.add_trace(
        go.Bar(
            name="Hallucination Rate (lower is better)",
            x=["Hallucination Rate"],
            y=[round(hallucination_rate, 4)],
            marker_color="#dc2626",
            text=[f"{hallucination_rate:.2%}"],
            textposition="outside",
        )
    )

    if retrieval_records:
        avg_recall = sum(r.recall_at_k for r in retrieval_records) / len(retrieval_records)
        avg_precision = sum(r.precision_at_k for r in retrieval_records) / len(retrieval_records)
        avg_mrr = sum(r.mrr for r in retrieval_records) / len(retrieval_records)

        fig.add_trace(
            go.Bar(
                name="Retrieval Metrics",
                x=["Recall@5", "Precision@5", "MRR"],
                y=[round(avg_recall, 4), round(avg_precision, 4), round(avg_mrr, 4)],
                marker_color="#16a34a",
                text=[f"{avg_recall:.2f}", f"{avg_precision:.2f}", f"{avg_mrr:.2f}"],
                textposition="outside",
            )
        )

    fig.update_layout(
        title=f"RAG Evaluation Dashboard ({len(evaluator.history)} queries evaluated)",
        yaxis=dict(title="Score", range=[0, 1.15]),
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
    )

    fig.write_html(output_path)
    logger.info("Dashboard saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    # Quick standalone demo using a tiny synthetic history
    import sys

    sys.path.insert(0, "src")
    from langchain.schema import Document

    logging.basicConfig(level=logging.INFO)

    demo_evaluator = RAGEvaluation()
    demo_docs = [Document(page_content="Demo content", metadata={"chunk_id": "c1"})]
    demo_evaluator.evaluate_query(
        query="demo query",
        answer="Demo grounded answer.",
        retrieved_docs=demo_docs,
        relevant_docs=demo_docs,
    )
    path = build_dashboard(demo_evaluator, "evaluation_dashboard.html")
    print(f"Dashboard written to {path}")
