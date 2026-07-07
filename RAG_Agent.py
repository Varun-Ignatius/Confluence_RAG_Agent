import logging
from langchain_ollama import OllamaLLM
from retriever import retrievel

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    """Build a formatted prompt from query and retrieved chunks."""
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
    """Retrieve relevant chunks and query the LLM to produce an answer."""
    logger.info("ask_agent called with query: %r", query)
    try:
        chunks = retrievel(query, n_results=3)
        if not chunks:
            logger.warning("No chunks retrieved for query: %r", query)
        else:
            logger.info("Retrieved %d chunk(s) for query.", len(chunks))

        prompt = build_prompt(query, chunks)

        logger.info("Invoking LLM...")
        response = llm.invoke(prompt)
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