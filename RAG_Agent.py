from langchain_ollama import OllamaLLM
from retriever import retrievel



llm = OllamaLLM(model="llama3.2")

SYSTEM_PROMPT = """You are a helpful assistant for internal Confluence documentation.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say: "I couldn't find relevant information in the documentation."
Be concise and cite which part of the context you used.
Context:
{context}
Question: {question}
Answer:"""

def build_prompt(query: str, chunks: list) -> str:
    context = "\n---\n".join(chunks)
    return SYSTEM_PROMPT.format(context=context, question=query)

def ask_agent(query:str):
    chunks = retrievel(query, n_results=3)
    prompt = build_prompt(query, chunks)
    response = llm.invoke(prompt)
    return response

if __name__ == "__main__":
    query = "Claims failed due to Member Not Found, what might be the root cause"
    response = ask_agent(query)
    print(response)
    
    
    