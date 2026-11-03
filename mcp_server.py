"""
mcp_server.py — MCP server exposing the Confluence RAG Agent as tools.

Lets other agents (e.g. a prod support agent) discover and call the runbook
knowledge base as tools over the Model Context Protocol.

Tools:
  search_runbooks  → raw relevant chunks + their Confluence sources (no LLM call);
                     preferred when the calling agent has its own capable LLM
  ask_runbook      → LLM-generated answer grounded in the runbooks + cited sources

Usage
-----
    python mcp_server.py                                # stdio (agent spawns this process)
    python mcp_server.py --transport streamable-http    # HTTP at http://127.0.0.1:8001/mcp

Environment variables (in addition to the ones the RAG pipeline needs):
    RAG_TOOL_TIMEOUT_SECONDS   per-call timeout for tool execution (default 120)
"""

import argparse
import asyncio
import logging
import os
from typing import Annotated

from langchain_core.documents import Document
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from RAG_Agent import NOT_FOUND_MESSAGE, ask_agent_with_documents
from retriever import retrieve_documents

# ── Logger ────────────────────────────────────────────────────────────────────
# logging writes to stderr by default, which keeps stdout clean for the stdio transport
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TOOL_TIMEOUT_SECONDS = float(os.getenv("RAG_TOOL_TIMEOUT_SECONDS", "120"))

# Tools only read the knowledge base; hints let clients auto-approve / retry safely
READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

KNOWLEDGE_BASE_SCOPE = (
    "the internal Confluence production-support runbooks for the revenue cycle / EDI domain: "
    "270/271 eligibility verification failures, 837 claim rejections from payers or the "
    "clearinghouse, and related registration and billing issues. Runbooks cover default "
    "severity and when to escalate (P2/P3), symptoms and impact, quick checks, root causes "
    "and how to confirm them (e.g. AAA / 999 / 277CA reject codes), workarounds, resolution "
    "steps, escalation contacts and timelines, and PHI/compliance rules"
)

# ── Server ────────────────────────────────────────────────────────────────────
mcp = MCPServer(
    name="confluence-runbooks",
    title="Confluence Runbooks",
    version="1.0.0",
    instructions=(
        f"Knowledge base of {KNOWLEDGE_BASE_SCOPE}. "
        "Use search_runbooks to get source excerpts you can reason over and cite; use "
        "ask_runbook when you want a ready-made answer. Always cite the returned source URLs. "
        "Results may contain PHI-handling guidance; never echo patient identifiers back."
    ),
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class Source(BaseModel):
    title: str = Field(description="Confluence page title.")
    url: str = Field(description="Confluence page URL — cite this in your response.")
    page_id: str | None = Field(default=None, description="Confluence page ID.")
    last_updated: str | None = Field(default=None, description="ISO timestamp the page was last updated.")


class Chunk(BaseModel):
    content: str = Field(description="Text excerpt from the runbook.")
    source: Source


class SearchResult(BaseModel):
    query: str = Field(description="The query echoed back.")
    found: bool = Field(description="False if no runbook content was retrieved.")
    chunks: list[Chunk] = Field(description="Relevant excerpts, most relevant first.")


class AnswerResult(BaseModel):
    question: str = Field(description="The question echoed back.")
    found: bool = Field(
        description="False if the runbooks contain nothing relevant; `answer` then carries no information."
    )
    answer: str = Field(description="Answer grounded only in the runbooks, one bullet per part of the question.")
    sources: list[Source] = Field(description="Distinct Confluence pages the answer was generated from.")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_source(doc: Document) -> Source:
    meta = doc.metadata or {}
    return Source(
        title=meta.get("title", "Untitled"),
        url=meta.get("source", ""),
        page_id=meta.get("id"),
        last_updated=meta.get("when"),
    )


def _unique_sources(docs: list[Document]) -> list[Source]:
    """One Source per Confluence page, in retrieval order."""
    seen: dict[str, Source] = {}
    for doc in docs:
        source = _to_source(doc)
        seen.setdefault(source.url or source.title, source)
    return list(seen.values())


async def _run_blocking(func, *args):
    """Run a blocking pipeline call off the event loop, bounded by a timeout.

    Raises ToolError so the calling agent gets a clean, actionable message
    (returned as an MCP tool error) instead of a stack trace.
    """
    try:
        return await asyncio.wait_for(asyncio.to_thread(func, *args), TOOL_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.error("%s timed out after %.0fs", func.__name__, TOOL_TIMEOUT_SECONDS)
        raise ToolError(
            f"The runbook knowledge base timed out after {TOOL_TIMEOUT_SECONDS:.0f}s. "
            "Retry once, or continue without runbook guidance."
        )
    except Exception as e:
        logger.error("%s failed: %s", func.__name__, e, exc_info=True)
        raise ToolError(
            "The runbook knowledge base is unavailable (retrieval or LLM backend error). "
            "Retry later, or continue without runbook guidance."
        )


# ── Tools ─────────────────────────────────────────────────────────────────────
# Descriptions are passed explicitly: the calling LLM uses them to decide when to
# call each tool (an f-string in place of a docstring would not become __doc__)
SEARCH_DESCRIPTION = f"""Search {KNOWLEDGE_BASE_SCOPE}.

Returns the most relevant runbook excerpts with their Confluence page title and URL,
without generating an answer. Prefer this tool: reason over the excerpts yourself and
cite the URLs. Excerpts may include content from a neighbouring runbook — ignore
excerpts whose source title does not match the issue."""

ASK_DESCRIPTION = f"""Answer a question using {KNOWLEDGE_BASE_SCOPE}.

Returns an LLM-generated answer grounded only in the runbooks, plus the Confluence pages
it came from. Slower than search_runbooks (it makes an extra LLM call) and uses a small
model, so verify critical details (severity, escalation contacts, PHI rules) against the
sources. If `found` is false, the runbooks do not cover the question."""


@mcp.tool(description=SEARCH_DESCRIPTION, annotations=READ_ONLY)
async def search_runbooks(
    query: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description=(
                "Natural-language description of the production issue or question, including "
                "concrete details such as transaction type (270/271, 837), reject codes, error "
                "messages, number of payers affected, and duration."
            ),
        ),
    ],
    n_results: Annotated[
        int,
        Field(ge=1, le=10, description="Number of excerpts to return."),
    ] = 5,
) -> SearchResult:
    """Retrieve runbook excerpts with sources (no LLM call)."""
    logger.info("search_runbooks | query=%r | n_results=%d", query, n_results)
    docs = await _run_blocking(retrieve_documents, query, n_results)
    return SearchResult(
        query=query,
        found=bool(docs),
        chunks=[Chunk(content=doc.page_content, source=_to_source(doc)) for doc in docs],
    )


@mcp.tool(description=ASK_DESCRIPTION, annotations=READ_ONLY)
async def ask_runbook(
    question: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2000,
            description=(
                "The question to answer from the runbooks. Multi-part questions are fine "
                "(e.g. 'What is the root cause and how should it be confirmed?'); each part is "
                "answered separately. Include concrete details such as reject codes and duration."
            ),
        ),
    ],
    n_results: Annotated[
        int,
        Field(ge=1, le=10, description="Number of runbook excerpts to ground the answer in."),
    ] = 5,
) -> AnswerResult:
    """Generate a grounded answer with cited sources."""
    logger.info("ask_runbook | question=%r | n_results=%d", question, n_results)
    answer, docs = await _run_blocking(ask_agent_with_documents, question, n_results)
    found = bool(docs) and NOT_FOUND_MESSAGE not in answer
    return AnswerResult(
        question=question,
        found=found,
        answer=answer,
        sources=_unique_sources(docs) if found else [],
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Confluence runbooks MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="stdio when the calling agent launches this process; streamable-http to run it as a service.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for streamable-http.")
    parser.add_argument("--port", type=int, default=8001, help="Port for streamable-http.")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run()
    else:
        logger.info("Serving MCP over streamable-http at http://%s:%d/mcp", args.host, args.port)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
