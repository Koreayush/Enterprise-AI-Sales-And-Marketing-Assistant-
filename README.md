# Enterprise AI Sales & Marketing Assistant

[![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)](https://shields.io)
[![Architecture](https://img.shields.io/badge/architecture-hybrid%20RAG-blue)](https://shields.io)
[![Evaluation](https://img.shields.io/badge/evaluation-faithfulness%20gated-purple)](https://shields.io)
[![License](https://img.shields.io/badge/license-Apache%202.0-orange)](LICENSE)

A production-ready **Generative AI assistant** for enterprise sales and marketing teams, built on a **hybrid RAG architecture** with automatic evaluation, source-grounded answers, and anti-hallucination guardrails.

## Overview

This system helps teams query company knowledge, generate personalized outreach, and monitor answer quality in real time. It combines keyword search, semantic retrieval, and reciprocal rank fusion to deliver grounded responses from your internal documents.

## Key Capabilities

- **Grounded Q&A** — Answer questions using only retrieved company documents, with citations.
- **Personalized email generation** — Create sales emails tailored to a customer’s name, company, pain points, and business context.
- **Live evaluation** — Score every response for faithfulness, relevance, and overall quality.
- **Hallucination protection** — Automatically reject unsupported responses before they reach the user.
- **Benchmarking and reporting** — Run evaluation tests and generate a Plotly dashboard for ongoing quality tracking.

## Why This Project Stands Out

Unlike a standard chatbot, this assistant is designed for enterprise reliability. It does not simply generate text; it validates answers, measures quality, and enforces a confidence gate to reduce hallucinations. That makes it suitable for sales, marketing, support, and internal knowledge workflows where accuracy matters.

## Architecture

```text
User Query
    │
    ▼
Hybrid Retriever
(BM25 + Vector Search + RRF)
    │
    ▼
Grounded LLM Generator
    │
    ▼
Evaluation Layer
(Faithfulness, Relevance, Quality)
    │
    ▼
Confidence Gate
    ├── Faithfulness < 0.5 → Fallback response
    └── Faithfulness ≥ 0.5 → Return answer
```

## Project Structure

```text
enterprise-ai-assistant/
├── data/                      # Source documents (.pdf, .txt, .csv)
├── src/
│   ├── document_loader.py     # Document ingestion
│   ├── text_processor.py      # Chunking and cleaning
│   ├── embedding_manager.py   # Embeddings + vector store
│   ├── retriever.py           # Hybrid retrieval with RRF
│   ├── llm_generator.py       # LLM wrapper
│   ├── prompt_templates.py    # Prompt definitions
│   └── rag_pipeline.py        # Retrieval-to-answer orchestration
├── evaluation.py              # Metrics, benchmarking, reporting
├── dashboard.py               # Plotly dashboard generator
├── main.py                   # FastAPI application
├── requirements.txt
├── .env.example
└── README.md
```

## Features at a Glance

| Capability | Description |
|---|---|
| Hybrid RAG | Combines BM25 keyword search with vector retrieval for stronger recall. |
| Reciprocal Rank Fusion | Merges retrieval results into a more robust ranked list. |
| Evaluation layer | Measures faithfulness, relevance, and overall quality for each query. |
| Hallucination gate | Replaces weak answers with a safe fallback message. |
| Email generation | Produces tailored outreach content from company knowledge. |
| Dashboard | Generates a Plotly-based HTML report for monitoring performance. |

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment variables

```bash
cp .env.example .env
```

Set one of the supported API keys:

```ini
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...
```

To use local embeddings instead of OpenAI embeddings:

```ini
EMBEDDING_PROVIDER=huggingface
```

### 3) Add documents

Place your `.pdf`, `.txt`, or `.csv` files inside the `data/` directory. Sample documents are included so the application works immediately after setup.

### 4) Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open the interactive API docs at:

```text
http://localhost:8000/docs
```

## API Endpoints

### `POST /chat/`

Ask a question about your internal documents.

**Request**
```json
{
  "query": "What is enterprise pricing?",
  "customer_context": "Healthcare client"
}
```

**Response**
```json
{
  "query": "What is enterprise pricing?",
  "response": "The Enterprise plan costs $499 per month per seat, billed annually...",
  "sources": ["company_docs.txt"],
  "confidence": 0.91,
  "evaluation": {
    "faithfulness": 0.95,
    "answer_relevance": 0.88,
    "overall_quality": 0.92,
    "is_hallucination": false
  }
}
```

### `POST /generate-email/`

Generate a personalized sales email.

**Request**
```json
{
  "customer_name": "Dr. Jane Smith",
  "company": "City Hospital",
  "pain_point": "slow patient onboarding workflows",
  "email_type": "cold outreach",
  "context": "Referred by a mutual contact"
}
```

### `POST /upload-doc/`

Upload new `.pdf`, `.txt`, or `.csv` documents and rebuild the index for consistent hybrid retrieval.

### `GET /evaluate/benchmark/`

Run a built-in benchmark suite and return aggregate evaluation results.

### `GET /evaluate/report/`

View a human-readable evaluation report for the current session.

### `POST /evaluate/save/`

Persist evaluation results and regenerate the dashboard.

### `GET /health/`

Check whether the pipeline is ready.

## Evaluation Strategy

| Metric | Meaning | Target |
|---|---|---|
| Recall@5 | Relevant documents found in the top 5 results | > 0.85 |
| Precision@5 | Fraction of top 5 results that are relevant | > 0.80 |
| MRR | Rank quality of the first relevant document | > 0.75 |
| Faithfulness | Answer grounded only in retrieved context | > 0.90 |
| Answer Relevance | Semantic similarity between query and answer | > 0.85 |
| Overall Quality | Combined quality score | > 0.70 |
| Hallucination Rate | Share of answers below faithfulness threshold | < 0.05 |

The anti-hallucination gate is enforced inside the pipeline, not only in the prompt. Any answer with faithfulness below `0.5` is replaced with:

```text
I don't have that information in our current documents.
```

## Validation Workflow

```bash
# Health check
curl http://localhost:8000/health/

# Ask a grounded question
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query":"What is enterprise pricing?"}'

# Test hallucination protection
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query":"What is your refund policy for a 10-year contract?"}'

# Generate a sales email
curl -X POST http://localhost:8000/generate-email/ \
  -H "Content-Type: application/json" \
  -d '{"customer_name":"Dr. Smith","company":"City Hospital","pain_point":"slow onboarding","email_type":"cold outreach"}'

# Run benchmark suite
curl http://localhost:8000/evaluate/benchmark/

# Save report and dashboard
curl -X POST http://localhost:8000/evaluate/save/
```

## Configuration

You can tune retrieval behavior through environment variables:

```ini
BM25_WEIGHT=0.4
VECTOR_WEIGHT=0.6
```

For keyword-heavy data such as product codes or exact terminology, increasing the BM25 weight may improve results.

## Notes

A few implementation details were adjusted for compatibility and reliability:

1. `rank_bm25` is used instead of the non-existent `bm25==1.6.4`.
2. `sentence-transformers` is optional and only required for local HuggingFace embeddings.
3. Sample content is included as `.txt` files, but the loader also supports PDFs.
4. `httpx==0.27.2` is pinned for compatibility with the selected OpenAI SDK version.
5. Minor ChromaDB telemetry warnings may appear in logs and do not affect functionality.

## Deployment

### Docker

```bash
docker build -t enterprise-ai-assistant .
docker run -p 8000:8000 --env-file .env enterprise-ai-assistant
```

### Direct

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Roadmap

- Add streaming chat responses.
- Introduce role-based access control.
- Expand evaluation into per-document and per-user analytics.
- Add support for more embedding and LLM providers.
- Provide downloadable PDF reports from the dashboard.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
