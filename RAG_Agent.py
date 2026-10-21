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
    # temperature=0 keeps answers deterministic so eval runs are comparable
    llm = OllamaLLM(model="llama3.2", temperature=0)
    logger.info("OllamaLLM model 'llama3.2' initialised successfully.")
except Exception as e:
    logger.critical("Failed to initialise OllamaLLM: %s", e, exc_info=True)
    raise

SYSTEM_PROMPT = """You are a helpful assistant for internal Confluence documentation.
Answer the user's question using ONLY the context provided below.

Instructions:
1. Questions often ask several things at once (e.g. "what is X and when should Y happen?",
   "what is the cause and how is it confirmed?", "who should be notified and what precautions apply?").
   Identify every part of the question and answer EACH part explicitly, one bullet per part,
   starting each bullet with a short label for the part it answers.
2. Include the specific facts from the context: severity levels, codes, time thresholds,
   team or role names, and step-by-step actions. Quote these values exactly as written.
3. Do not refer to the context by position (e.g. "see section 8" or "point 3");
   state the actual content instead.
4. Ignore context that is about a different system or issue type than the question.
5. The context is often a flattened table: a row label (e.g. "All payers failing") is followed
   by its details in the same line. Match the situation in the question to these row labels.
6. Only if, after checking every chunk, a part of the question is truly not covered, write
   "The documentation does not cover this." as the answer to that part. Never add it as an extra bullet.
   If nothing in the context is relevant, say: "I couldn't find relevant information in the documentation."

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
    """Retrieve relevant chunks and query the LLM; returns only the answer."""
    answer, _ = ask_agent_with_context(query)
    return answer


def ask_agent_with_context(query: str) -> tuple[str, list[str]]:
    """Retrieve relevant chunks and query the LLM.

    Returns (answer, chunks) so callers such as the eval harness can judge the
    answer against exactly the context the LLM saw.

    Each call creates a Langfuse trace that captures:
      - the retrieval span (via @observe in retriever.py)
      - the full LLM call (inputs, output, latency, model name)
    """
    logger.info("ask_agent called with query: %r", query)
    try:
        # --- Retrieval ---
        chunks = retrievel(query, n_results=5)
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
        return response, chunks

    except Exception as e:
        logger.error("Error in ask_agent for query %r: %s", query, e, exc_info=True)
        raise


if __name__ == "__main__":
    query = "What is the default severity of an eligibility verification failure involving 270/271 EDI transactions, and when should it be escalated to P2?"
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