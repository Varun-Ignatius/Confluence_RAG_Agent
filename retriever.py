import logging
import os

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

chroma_api_key = os.getenv("chroma_api_key")
chroma_tenant = os.getenv("chroma_tenant")
chroma_database = os.getenv("chroma_database")

if not chroma_api_key:
    logger.warning("Environment variable 'chroma_api_key' is not set.")
if not chroma_tenant:
    logger.warning("Environment variable 'chroma_tenant' is not set.")
if not chroma_database:
    logger.warning("Environment variable 'chroma_database' is not set.")

# ── ChromaDB client ───────────────────────────────────────────────────────────
try:
    chroma_Client = chromadb.CloudClient(
        api_key=chroma_api_key,
        tenant=chroma_tenant,
        database=chroma_database,
    )
    logger.info("ChromaDB CloudClient initialised successfully.")
except Exception as e:
    logger.critical("Failed to initialise ChromaDB CloudClient: %s", e, exc_info=True)
    raise

# ── Embeddings ────────────────────────────────────────────────────────────────
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    logger.info("OllamaEmbeddings model 'nomic-embed-text' initialised successfully.")
except Exception as e:
    logger.critical("Failed to initialise OllamaEmbeddings: %s", e, exc_info=True)
    raise

# ── Vector store ──────────────────────────────────────────────────────────────
try:
    vectorstore = Chroma(
        client=chroma_Client,
        collection_name="confluence_docs",
        embedding_function=embeddings,
    )
    logger.info("Chroma vectorstore connected to collection 'confluence_docs'.")
except Exception as e:
    logger.critical("Failed to create Chroma vectorstore: %s", e, exc_info=True)
    raise


# ── Retrieval function ────────────────────────────────────────────────────────
def retrievel(query: str, n_results: int = 5) -> list[str]:
    """Retrieve the top-n relevant document chunks for the given query."""
    logger.info("retrievel called | query=%r | n_results=%d", query, n_results)
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
        docs = retriever.invoke(query)
        logger.info("Retrieved %d document(s) from vectorstore.", len(docs))
        return [doc.page_content for doc in docs]
    except Exception as e:
        logger.error(
            "Error retrieving documents for query %r: %s", query, e, exc_info=True
        )
        raise


if __name__ == "__main__":
    pass  # retrieval is invoked from RAG_Agent.py
