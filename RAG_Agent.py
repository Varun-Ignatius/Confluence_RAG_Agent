import logging
from langchain_ollama import OllamaLLM
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from retriever import retrievel

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Langfuse callback handler (reads keys from .env automatically) ─────────────
# Requires LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env
try:
    langfuse_handler = LangfuseCallbackHandler()
    logger.info("Langfuse CallbackHandler initialised — traces will be sent to Langfuse.")
except Exception as e:
    langfuse_handler = None
    logger.warning(
        "Langfuse CallbackHandler could not be initialised (%s). "
        "Tracing will be disabled; set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env.",
        e,
    )

# ── LLM ───────────────────────────────────────────────────────────────────────
try:
    llm = OllamaLLM(model="llama3.2")
    logger.info("OllamaLLM model 'llama3.2' initialised successfully.")
except Exception as e:
    logger.critical("Failed to initialise OllamaLLM: %s", e, exc_info=True)
    raise

SYSTEM_PROMPT = """You are a helpful assistant for internal Confluence documentation.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say: "I couldn't find relevant information in the documentation."
Be concise and cite which part of the context you used.
Context:
{context}
Question: {question}
Answer:"""


def build_prompt(query: str, chunks: list) -> str:
    """Build a formatted prompt from query and retrieved context chunks."""
    try:
        logger.debug("Building prompt for query: %r with %d chunk(s).", query, len(chunks))
        context = "\n---\n".join(chunks)
        prompt = SYSTEM_PROMPT.format(context=context, question=query)
        logger.debug("Prompt built successfully.")
        return prompt
    except KeyError as e:
        logger.error("Prompt template is missing a required key: %s", e, exc_info=True)
        raise
    except Exception as e:
        logger.error("Unexpected error while building prompt: %s", e, exc_info=True)
        raise


def ask_agent(query: str) -> str:
    """Retrieve relevant chunks and query the LLM.

    Each call creates a Langfuse trace that captures:
      - the retrieval span (via @observe in retriever.py)
      - the full LLM call (inputs, output, latency, model name)
    """
    logger.info("ask_agent called with query: %r", query)
    try:
        # --- Retrieval ---
        chunks = retrievel(query, n_results=3)
        if not chunks:
            logger.warning("No chunks retrieved for query: %r", query)
        else:
            logger.info("Retrieved %d chunk(s) for query.", len(chunks))

        # --- Prompt ---
        prompt = build_prompt(query, chunks)

        # --- LLM call — pass Langfuse handler so the call is traced ---
        callbacks = [langfuse_handler] if langfuse_handler else []
        logger.info("Invoking LLM (Langfuse tracing: %s)...", "ON" if callbacks else "OFF")
        response = llm.invoke(prompt, config={"callbacks": callbacks})
        logger.info("LLM responded successfully.")
        return response

    except Exception as e:
        logger.error("Error in ask_agent for query %r: %s", query, e, exc_info=True)
        raise


if __name__ == "__main__":
    query = "Claims failed due to Member Not Found, what might be the root cause"
    try:
        response = ask_agent(query)
        print(response)
    except Exception as e:
        logger.critical("Unhandled exception in main: %s", e, exc_info=True)
    finally:
        # Flush buffered traces to Langfuse before the process exits
        if langfuse_handler:
            try:
                langfuse_handler.flush()
                logger.info("Langfuse traces flushed.")
            except Exception as flush_err:
                logger.warning("Failed to flush Langfuse traces: %s", flush_err)