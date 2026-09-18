"""
RAG engine: retrieves relevant chunks from the vector store 
and generates an answer using a local Ollama model.
"""
import chromadb
from chromadb.utils import embedding_functions
import requests

VECTORSTORE_DIR = "data/vectorstore"
import os
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "llama3.2:1b"
TOP_K = 3  # how many chunks to retrieve

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_collection():
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    return client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn
    )


def retrieve(question, top_k=TOP_K):
    """Find the most relevant chunks for a question."""
    collection = get_collection()
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    chunks = results["documents"][0]
    sources = [meta["source"] for meta in results["metadatas"][0]]
    return chunks, sources


def build_prompt(question, chunks):
    """Combine retrieved chunks into a prompt for the LLM."""
    context = "\n\n".join(f"[Excerpt {i+1}]: {c}" for i, c in enumerate(chunks))
    prompt = f"""You are a helpful assistant that answers questions using ONLY the context provided below. If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def generate_answer(prompt):
    """Send prompt to local Ollama model and get response."""
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    response.raise_for_status()
    return response.json()["response"]


def answer_question(question):
    """Full RAG pipeline: retrieve, then generate."""
    chunks, sources = retrieve(question)
    if not chunks:
        return "No documents found. Please run ingest.py first.", []
    prompt = build_prompt(question, chunks)
    answer = generate_answer(prompt)
    return answer, sources


if __name__ == "__main__":
    print("RAG Chatbot (type 'quit' to exit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer, sources = answer_question(question)
        print(f"\nBot: {answer}")
        print(f"Sources: {', '.join(set(sources))}\n")