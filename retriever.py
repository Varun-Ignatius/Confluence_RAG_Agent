import chromadb
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv() 

chroma_api_key = os.getenv("chroma_api_key")
chroma_tenant = os.getenv("chroma_tenant")
chroma_database = os.getenv("chroma_database")
chroma_Client = chromadb.CloudClient(
  api_key=chroma_api_key,
  tenant=chroma_tenant,
  database=chroma_database
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(client=chroma_Client, collection_name= "confluence_docs", embedding_function=embeddings)

def retrievel(query: str, n_results: int = 5):
    retriever = vectorstore.as_retriever(search_kwargs={"k": n_results})
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


if __name__ == "__main__":
    query = "I've recieved error code ERR-EDI-837-VAL, what might be the root cause"
    results = retrievel(query)
    print(results)



