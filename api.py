"""
api.py — FastAPI wrapper for the Confluence RAG Agent.

Exposes the ask_agent() function as a REST API so any other agent or UI
can query the Confluence knowledge base over HTTP.

Endpoints:
  GET  /health       → liveness probe
  POST /ask          → query the RAG agent (JSON body: { "query": "..." })
"""

import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from RAG_Agent import ask_agent

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Confluence RAG Agent API",
    description=(
        "Query your internal Confluence documentation using a RAG pipeline "
        "(ChromaDB + Ollama llama3.2). Designed to be consumed by other AI "
        "agents or UI applications."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# ── CORS — allow any origin so UI apps / agents can call freely ───────────────
# Restrict origins in production by replacing ["*"] with your allowed domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The natural-language question to ask about Confluence docs.",
        examples=["Claims failed due to Member Not Found, what might be the root cause?"],
    )
    n_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve from the vector store.",
    )


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The LLM-generated answer.")
    query: str = Field(..., description="The original query echoed back.")
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds.")


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["Ops"],
)
async def health():
    """Returns 200 OK when the service is up. Use this as a liveness/readiness probe."""
    return HealthResponse(status="ok", version=app.version)


@app.post(
    "/ask",
    response_model=QueryResponse,
    summary="Query the Confluence RAG Agent",
    tags=["RAG"],
)
async def ask(request: QueryRequest):
    """
    Send a natural-language question and receive an answer grounded in your
    Confluence documentation.

    The pipeline:
    1. Embeds the query with `nomic-embed-text` (Ollama)
    2. Retrieves top-N chunks from ChromaDB Cloud
    3. Feeds context + question to `llama3.2` (Ollama)
    4. Returns the answer (all calls traced in Langfuse)
    """
    logger.info("POST /ask | query=%r | n_results=%d", request.query, request.n_results)
    start = time.perf_counter()

    try:
        answer = ask_agent(request.query)
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("POST /ask | completed in %.1f ms", latency_ms)
        return QueryResponse(
            answer=answer,
            query=request.query,
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        logger.error("POST /ask | error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
