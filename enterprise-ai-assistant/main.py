"""
main.py
-------
FastAPI entry point for the Enterprise AI Sales & Marketing Assistant.

Endpoints:
  POST /chat/                  - grounded query answering with live evaluation
  POST /generate-email/        - personalized email generation
  POST /upload-doc/            - upload + index a new document
  GET  /evaluate/benchmark/    - run the benchmark test suite
  GET  /evaluate/report/       - human-readable evaluation report
  POST /evaluate/save/         - persist evaluation_report.json
  GET  /health/                - health check
  GET  /docs                   - Swagger UI (provided automatically by FastAPI)
"""

import logging
import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Make src/ importable before local imports for runtime
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.document_loader import DocumentLoader, DocumentLoadError
from src.embedding_manager import EmbeddingManager
from evaluation import RAGEvaluation
from src.llm_generator import LLMGenerator
from src.rag_pipeline import HybridRAGPipeline
from src.retriever import HybridRetriever
from src.text_processor import TextProcessor
from dashboard import build_dashboard

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("main")

DATA_DIR = os.environ.get("DATA_DIR", "data")
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
COLLECTION_NAME = "enterprise_docs_hybrid"
DEFAULT_K = 5
BM25_WEIGHT = float(os.environ.get("BM25_WEIGHT", 0.4))
VECTOR_WEIGHT = float(os.environ.get("VECTOR_WEIGHT", 0.6))


# ----------------------------------------------------------------------
# Application state - holds the live pipeline + evaluator
# ----------------------------------------------------------------------
class AppState:
    pipeline: Optional[HybridRAGPipeline] = None
    evaluator: Optional[RAGEvaluation] = None
    embedding_manager: Optional[EmbeddingManager] = None
    documents: List = []  # all currently indexed chunks (needed to rebuild BM25 on upload)


state = AppState()


def build_pipeline() -> None:
    """
    (Re)build the full pipeline from whatever documents currently exist in
    DATA_DIR: load -> chunk -> embed -> hybrid retriever -> LLM -> evaluator.
    Raises a clear RuntimeError if no documents are available or no API
    key is configured, rather than failing silently.
    """
    loader = DocumentLoader(data_dir=DATA_DIR)
    raw_docs = loader.load_directory()
    if not raw_docs:
        raise RuntimeError(
            f"No documents found in '{DATA_DIR}'. Add at least one .pdf/.txt/.csv file "
            "and restart, or use POST /upload-doc/."
        )

    processor = TextProcessor(chunk_size=500, chunk_overlap=50)
    chunks = processor.split_documents(raw_docs)

    embedding_manager = EmbeddingManager(
        collection_name=COLLECTION_NAME, persist_directory=PERSIST_DIR
    )
    vectorstore = embedding_manager.build_vectorstore(chunks)

    hybrid_retriever = HybridRetriever(
        documents=chunks, vectorstore=vectorstore, weights=(BM25_WEIGHT, VECTOR_WEIGHT), k=DEFAULT_K
    )

    llm_generator = LLMGenerator()
    evaluator = RAGEvaluation(llm_generator=llm_generator, embeddings=embedding_manager.embeddings)

    pipeline = HybridRAGPipeline(
        retriever=hybrid_retriever, llm_generator=llm_generator, evaluator=evaluator, k=DEFAULT_K
    )

    state.pipeline = pipeline
    state.evaluator = evaluator
    state.embedding_manager = embedding_manager
    state.documents = chunks

    logger.info("Pipeline built successfully: %d chunks indexed.", len(chunks))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        build_pipeline()
    except Exception as exc:  # noqa: BLE001
        # Don't crash app startup - allow /health and /upload-doc/ to work,
        # and surface a clear error on endpoints that need the pipeline.
        logger.error("Pipeline initialization failed at startup: %s", exc)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Enterprise AI Sales & Marketing Assistant",
    description="Hybrid RAG (BM25 + Vector + RRF) chat assistant with anti-hallucination evaluation layer.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve the frontend
FRONTEND_DIR = Path(__file__).parent / "frontend"


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/app/index.html")


app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


def require_pipeline() -> HybridRAGPipeline:
    if state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG pipeline is not initialized. Ensure documents exist in the data "
                "directory and a valid OPENAI_API_KEY/GROQ_API_KEY is set in .env, "
                "then restart the server (or call POST /upload-doc/ to bootstrap)."
            ),
        )
    return state.pipeline


# ----------------------------------------------------------------------
# Request / response schemas
# ----------------------------------------------------------------------
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    customer_context: Optional[str] = Field(
        default="", description="Optional context about the customer, e.g. 'Healthcare client'."
    )


class ChatResponse(BaseModel):
    query: str
    response: str
    sources: List[str]
    confidence: Optional[float]
    evaluation: dict


class EmailRequest(BaseModel):
    customer_name: str
    company: str
    pain_point: str
    email_type: str = Field(..., description="e.g. 'cold outreach', 'follow-up', 'renewal'.")
    context: Optional[str] = Field(default="", description="Any additional notes.")


class EmailResponse(BaseModel):
    email_type: str
    content: str


class UploadResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    retrieval: str
    fusion: str
    pipeline: str


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answer a query using hybrid RAG retrieval + grounded generation.
    Every query is evaluated in real time (faithfulness, relevance, quality).
    If faithfulness < 0.5, the answer is rejected and replaced with the
    standard "I don't have that information" response.
    """
    pipeline = require_pipeline()
    try:
        result = pipeline.answer_query_with_evaluation(
            query=request.query, customer_context=request.customer_context or ""
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error handling /chat/ request")
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {exc}") from exc


@app.post("/generate-email/", response_model=EmailResponse)
async def generate_email(request: EmailRequest):
    """Generate a personalized, company-data-grounded sales email."""
    pipeline = require_pipeline()
    try:
        content = pipeline.generate_email(
            customer_name=request.customer_name,
            company=request.company,
            pain_point=request.pain_point,
            email_type=request.email_type,
            context=request.context or "",
        )
        return EmailResponse(email_type=request.email_type, content=content)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error handling /generate-email/ request")
        raise HTTPException(status_code=500, detail=f"Email generation failed: {exc}") from exc


@app.post("/upload-doc/", response_model=UploadResponse)
async def upload_doc(file: UploadFile = File(...)):
    """
    Upload a new document (.pdf, .txt, or .csv), index it, and add it to
    the live vector store + BM25 corpus (full pipeline rebuild to keep
    BM25 consistent, since BM25Retriever has no incremental add API).
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".csv"}:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type '{suffix}'. Use .pdf, .txt, or .csv."
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    dest_path = Path(DATA_DIR) / file.filename

    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    try:
        loader = DocumentLoader(data_dir=DATA_DIR)
        raw_docs = loader.load_file(str(dest_path))
        processor = TextProcessor(chunk_size=500, chunk_overlap=50)
        new_chunks = processor.split_documents(raw_docs)

        # Rebuild the full pipeline so BM25 + vector store both reflect the new doc.
        # (BM25Retriever in LangChain has no incremental-add API, so a full
        # rebuild from disk is the correct/safe approach here.)
        build_pipeline()

        return UploadResponse(
            status="success",
            filename=file.filename,
            chunks_created=len(new_chunks),
            message=f"Document indexed and pipeline rebuilt with {len(state.documents)} total chunks.",
        )
    except DocumentLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error handling /upload-doc/ request")
        raise HTTPException(status_code=500, detail=f"Upload/indexing failed: {exc}") from exc


@app.get("/evaluate/benchmark/")
async def evaluate_benchmark():
    """
    Run the benchmark test suite (3-5 known query/answer pairs) against
    the live pipeline. PASS if avg_overall_quality > 0.7 and avg_faithfulness > 0.8.
    """
    pipeline = require_pipeline()
    if state.evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized.")

    try:
        results = state.evaluator.run_benchmark_test(pipeline)
        return results
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error running benchmark")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {exc}") from exc


@app.get("/evaluate/report/")
async def evaluate_report():
    """Return a human-readable evaluation report summarizing all queries evaluated so far."""
    if state.evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized.")
    report_text = state.evaluator.generate_report()
    return {"report": report_text}


@app.post("/evaluate/save/")
async def evaluate_save():
    """Persist the current evaluation history to evaluation_report.json, and regenerate the dashboard."""
    if state.evaluator is None:
        raise HTTPException(status_code=503, detail="Evaluator not initialized.")
    try:
        report_path = state.evaluator.save_report("evaluation_report.json")
        dashboard_path = build_dashboard(state.evaluator, "evaluation_dashboard.html")
        return {
            "message": "Report saved",
            "report_path": report_path,
            "dashboard_path": dashboard_path,
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error saving evaluation report")
        raise HTTPException(status_code=500, detail=f"Save failed: {exc}") from exc


@app.get("/health/", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    pipeline_status = "Ready" if state.pipeline is not None else "Not initialized"
    return HealthResponse(
        status="ok",
        retrieval="Hybrid (BM25 + Vector)",
        fusion="RRF",
        pipeline=pipeline_status,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
