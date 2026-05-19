import os
import requests
from dotenv import load_dotenv
from groq import Groq
import chromadb

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")

HF_API_KEY = os.getenv("HF_API_KEY")
HF_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def get_embedding(text: str) -> list:
    response = requests.post(
        HF_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    return response.json()

def get_relevant_chunks(question: str, collection_name: str, n_results: int = 3) -> list:
    collection = chroma_client.get_or_create_collection(name=collection_name)
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"]
        })
    return chunks

def generate_answer(question: str, chunks: list, chat_history: list = []) -> str:
    context = ""
    for chunk in chunks:
        context += f"[Page {chunk['page']}]\n{chunk['text']}\n\n"

    system_prompt = f"""You are a helpful assistant for answering questions about a document.
Answer the question based only on the context below.
If the answer is not in the context, say "I could not find the answer in the document."

Context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]

    for msg in chat_history:
        if msg.get("role") in ["user", "assistant"] and msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    return response.choices[0].message.content

def ask(question: str, collection_name: str, chat_history: list = []) -> dict:
    chunks = get_relevant_chunks(question, collection_name)
    answer = generate_answer(question, chunks, chat_history)

    return {
        "question": question,
        "answer": answer,
        "sources": [f"Page {chunk['page']}" for chunk in chunks]
    }