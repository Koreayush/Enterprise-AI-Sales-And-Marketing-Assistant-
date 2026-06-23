# Enterprise AI Sales & Marketing Assistant

A production-ready Generative AI chat assistant built on a **hybrid RAG
architecture** (BM25 + Vector + Reciprocal Rank Fusion), with a built-in
**evaluation layer** that scores every answer for faithfulness and rejects
unsupported (hallucinated) responses automatically.

## What it does

- **Query answering** — ask questions about your company's documents and get
  answers grounded *only* in retrieved content, with source citations.
- **Personalized email generation** — generate sales emails tailored to a
  customer's name, company, and pain point, using real company data
  (pricing, ROI, features) pulled from your documents.
- **Evaluation on every query** — faithfulness, answer relevance, and overall
  quality scores are computed live and returned in the API response.
- **Anti-hallucination gate** — if an answer's faithfulness score falls below
  0.5, it is automatically discarded and replaced with
  `"I don't have that information in our current documents."` before it ever
  reaches the user.
- **Benchmark + dashboard** — a one-command benchmark test suite and a Plotly
  HTML dashboard for monitoring retrieval/generation quality over time.

## Architecture

```
Query
  │
  ▼
┌─────────────────────────────┐
│   Hybrid Retriever           │
│   BM25 (keyword, weight 0.4) │──┐
│   Vector (semantic, w. 0.6)  │──┼─► EnsembleRetriever (RRF) ─► top-k docs
└─────────────────────────────┘  │
                                  │
                                  ▼
                       ┌──────────────────┐
                       │  LLM Generator    │
                       │  (grounded answer)│
                       └─────────┬────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │  Evaluation Layer │
                       │  faithfulness,    │
                       │  relevance,       │
                       │  quality          │
                       └─────────┬────────┘
                                 │
                  faithfulness < 0.5?
                       │YES            │NO
                       ▼               ▼
              "I don't have      Return generated
               that information"      answer
```

## Project structure

```
enterprise-ai-assistant/
├── data/                      # Drop .pdf / .txt / .csv source documents here
│   ├── company_docs.txt       # Sample data (pricing, ROI, healthcare, security)
│   └── product_manual.txt     # Sample data (features, onboarding, integrations)
├── src/
│   ├── document_loader.py     # PDF / TXT / CSV ingestion
│   ├── text_processor.py      # Chunking (500 chars / 50 overlap) + cleaning
│   ├── embedding_manager.py   # OpenAI or HuggingFace embeddings -> ChromaDB
│   ├── retriever.py           # BM25 + Vector hybrid search with RRF
│   ├── llm_generator.py       # OpenAI or Groq LLM wrapper
│   ├── prompt_templates.py    # Query / email / faithfulness-judge prompts
│   └── rag_pipeline.py        # Orchestrates retrieval -> generation -> gating
├── evaluation.py              # RAGEvaluation: metrics, benchmark, reporting
├── dashboard.py                # Plotly evaluation_dashboard.html generator
├── main.py                    # FastAPI server (all endpoints)
├── requirements.txt
├── .env.example                # Copy to .env and fill in your API key
└── README.md
```

> **Note on sample data:** the spec's example structure references
> `pricing.pdf` / `product_manual.pdf`. The bundled samples are `.txt` files
> with equivalent content — the loader fully supports PDFs too
> (`PyPDFLoader`); drop your own `.pdf` files into `data/` and they'll be
> picked up automatically.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> The spec's `requirements.txt` listed `bm25==1.6.4`, which does not exist on
> PyPI — the correct package backing LangChain's `BM25Retriever` is
> `rank_bm25`, which is what's pinned here instead.

### 2. Configure your API key

```bash
cp .env.example .env
```

Edit `.env` and set **one** of:

```ini
OPENAI_API_KEY=sk-...      # if LLM_PROVIDER=openai (default)
GROQ_API_KEY=gsk_...       # if LLM_PROVIDER=groq (free alternative)
```

By default both the LLM and the embeddings use OpenAI. To run embeddings
locally for free instead (no OpenAI key needed for embeddings), set:

```ini
EMBEDDING_PROVIDER=huggingface
```

and `pip install sentence-transformers` (not installed by default — it's a
large dependency pulling in PyTorch).

### 3. Add your documents

Drop `.pdf`, `.txt`, or `.csv` files into `data/`. Two samples are included
so the app works out of the box.

### 4. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On startup, the app automatically loads `data/`, chunks it, builds the
ChromaDB vector store, and initializes the hybrid retriever. If no API key
is set or no documents are found, the server still starts (so `/health/`
always works), but `/chat/` and `/generate-email/` return a `503` with a
clear explanation until the pipeline is ready.

Visit **http://localhost:8000/docs** for interactive Swagger documentation.

## API Reference

### `POST /chat/`
```json
// Request
{ "query": "What is enterprise pricing?", "customer_context": "Healthcare client" }

// Response
{
  "query": "What is enterprise pricing?",
  "response": "The Enterprise plan costs $499 per month per seat, billed annually...",
  "sources": ["company_docs.txt"],
  "confidence": 0.91,
  "evaluation": {
    "query": "...",
    "timestamp": "...",
    "generation": {
      "faithfulness": 0.95,
      "answer_relevance": 0.88,
      "overall_quality": 0.92,
      "is_hallucination": false
    }
  }
}
```

### `POST /generate-email/`
```json
// Request
{
  "customer_name": "Dr. Jane Smith",
  "company": "City Hospital",
  "pain_point": "slow patient onboarding workflows",
  "email_type": "cold outreach",
  "context": "Referred by a mutual contact"
}

// Response
{ "email_type": "cold outreach", "content": "Dear Dr. Smith, ..." }
```

### `POST /upload-doc/`
Multipart file upload (`.pdf`, `.txt`, `.csv`). Re-indexes the full corpus
(BM25 has no incremental-add API, so the pipeline is rebuilt from disk to
keep BM25 and the vector store consistent).

### `GET /evaluate/benchmark/`
Runs a 5-case benchmark suite against the live pipeline:
```json
{ "status": "PASS", "avg_overall_quality": 0.81, "avg_faithfulness": 0.93, ... }
```

### `GET /evaluate/report/`
Human-readable summary of every query evaluated so far this session.

### `POST /evaluate/save/`
Persists `evaluation_report.json` and regenerates `evaluation_dashboard.html`.

### `GET /health/`
```json
{ "status": "ok", "retrieval": "Hybrid (BM25 + Vector)", "fusion": "RRF", "pipeline": "Ready" }
```

## Evaluation layer details

| Metric | Threshold | Meaning |
|---|---|---|
| Recall@5 | > 0.85 | Fraction of relevant docs found in top-5 |
| Precision@5 | > 0.80 | Fraction of top-5 docs that are relevant |
| MRR | > 0.75 | How high the first relevant doc ranks |
| Faithfulness | > 0.90 | Is the answer grounded *only* in retrieved context? (LLM-judged) |
| Answer Relevance | > 0.85 | Embedding cosine similarity between query and answer |
| Overall Quality | > 0.70 | 0.6×faithfulness + 0.4×relevance |
| Hallucination Rate | < 0.05 | Fraction of answers with faithfulness < 0.5 |

**The anti-hallucination gate is non-negotiable**: any answer scoring below
0.5 faithfulness is replaced with the standard "no information" response
*before* it is returned to the user — this happens inside
`HybridRAGPipeline.answer_query_with_evaluation()`, not just in the prompt.

If `evaluation.py` has no LLM/embeddings configured (e.g. for offline unit
testing), it falls back to conservative word-overlap heuristics rather than
failing — but production deployments should always run with real
LLM-judged faithfulness for accuracy.

## Testing it works

```bash
# 1. Health check
curl http://localhost:8000/health/

# 2. Ask a known-good question
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is enterprise pricing?"}'

# 3. Ask about something NOT in the docs (anti-hallucination test)
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your refund policy for a 10-year contract?"}'
# Expect: "I don't have that information in our current documents."

# 4. Generate an email
curl -X POST http://localhost:8000/generate-email/ \
  -H "Content-Type: application/json" \
  -d '{"customer_name":"Dr. Smith","company":"City Hospital","pain_point":"slow onboarding","email_type":"cold outreach"}'

# 5. Run the benchmark suite
curl http://localhost:8000/evaluate/benchmark/

# 6. Save report + dashboard
curl -X POST http://localhost:8000/evaluate/save/
open evaluation_dashboard.html
```

## Tuning retrieval weights

`BM25_WEIGHT` / `VECTOR_WEIGHT` in `.env` (default `0.4` / `0.6`, more
semantic-weighted). For keyword-heavy data (e.g. product SKUs, exact terms),
try `0.6` / `0.4` instead, then re-run `/evaluate/benchmark/` to compare.

## Known deviations from the original spec

These were necessary corrections, made explicit rather than silently
"fixed":

1. **`bm25==1.6.4`** in the spec's requirements.txt does not exist on PyPI.
   Replaced with `rank_bm25==0.2.2`, the actual package LangChain's
   `BM25Retriever` depends on.
2. **`sentence-transformers`** is listed as a dependency but is *not*
   installed by default in `requirements.txt`'s install path here, since it
   pulls in PyTorch (multi-GB). It's only required if you set
   `EMBEDDING_PROVIDER=huggingface`. OpenAI embeddings are used by default
   and don't need it.
3. Sample data is provided as `.txt` rather than `.pdf` (the loader fully
   supports PDFs — just drop your own in).
4. **`openai==1.12.0`** (as pinned in the original spec) is incompatible with
   modern `httpx` releases (`Client.__init__() got an unexpected keyword
   argument 'proxies'`). Fixed by pinning `httpx==0.27.2` alongside it in
   `requirements.txt`.
5. You may see harmless log lines like
   `Failed to send telemetry event ... capture() takes 1 positional argument`
   on startup — this is `chromadb==0.4.22`'s anonymized telemetry hitting a
   version mismatch with a newer `posthog` package. It does not affect
   functionality and can be safely ignored.

## Deployment

```bash
# Docker (optional)
docker build -t enterprise-ai-assistant .
docker run -p 8000:8000 --env-file .env enterprise-ai-assistant

# Or directly
uvicorn main:app --host 0.0.0.0 --port 8000
```
