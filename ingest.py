import os
import requests
import fitz
import chromadb
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def extract_text_from_pdf(pdf_path: str) -> list:
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text
            })

    doc.close()
    return pages


def split_into_chunks(pages: list, chunk_size: int = 1000, overlap: int = 200) -> list:
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_num = page["page"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": text[start:end]
            })
            chunk_id += 1
            start = end - overlap

    return chunks


def get_embedding(text: str) -> list:
    response = requests.post(
        HF_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    return response.json()


def embed_chunks(chunks: list) -> list:
    texts = [chunk["text"] for chunk in chunks]

    response = requests.post(
        HF_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )

    embeddings = response.json()

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]

    return chunks


chroma_client = chromadb.PersistentClient(path="./chroma_db")

def store_in_chromadb(chunks: list, collection_name: str) -> None:
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    collection = chroma_client.get_or_create_collection(name=collection_name)

    collection.add(
        ids=[str(chunk["chunk_id"]) for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{"page": chunk["page"]} for chunk in chunks]
    )