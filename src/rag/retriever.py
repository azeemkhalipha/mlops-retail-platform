import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv("/Users/azeemkhalipha/mlops-retail-platform/.env")

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/Users/azeemkhalipha/mlops-retail-platform")

# Add project root to path so src.rag.indexer can be found
sys.path.insert(0, PROJECT_ROOT)

from src.rag.indexer import get_collection, build_index  # noqa: E402

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def check_ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def retrieve_context(question: str, n_results: int = 4) -> list:
    try:
        collection = get_collection()
        if collection.count() == 0:
            collection = build_index()
    except Exception:
        collection = build_index()

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count())
    )
    return results["documents"][0] if results["documents"] else []


def ask(question: str) -> str:
    if not check_ollama_running():
        return "Ollama is not running. Start it with: ollama serve"

    context_docs = retrieve_context(question)
    if not context_docs:
        return "No relevant context found. Try rebuilding the index."

    context = "\n\n---\n\n".join(context_docs)

    prompt = f"""You are an MLOps assistant for a retail demand forecasting platform.
Answer the question based only on the context below.
Be concise and direct. Only use numbers and facts from the context.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""

    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=60)

    if response.status_code != 200:
        return f"Ollama error: {response.status_code}"

    return response.json().get("response", "No response generated").strip()


if __name__ == "__main__":
    print("Testing RAG pipeline...\n")

    if not check_ollama_running():
        print("Ollama not running. Start with: ollama serve")
    else:
        questions = [
            "Which model performed best?",
            "Should I retrain the model right now?",
            "Which features drifted the most?",
            "What does qty_lag_7 mean?",
            "What is the current drift share?"
        ]
        for q in questions:
            print(f"Q: {q}")
            print(f"A: {ask(q)}")
            print("-" * 60)
