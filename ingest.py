import os
import requests
import fitz  # this is PyMuPDF
#from sentence_transformers import SentenceTransformer
import chromadb

def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Opens a PDF and extracts text page by page.
    Returns a list like:
    [
        {"page": 1, "text": "..."},
        {"page": 2, "text": "..."},
    ]
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()        # extract raw text from this page

        if text.strip():              # skip empty pages
            pages.append({
                "page": page_num + 1, # page numbers start from 1
                "text": text
            })

    doc.close()
    return pages


def split_into_chunks(pages: list, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Takes pages and splits text into overlapping chunks.
    Returns a list like:
    [
        {"chunk_id": 0, "page": 1, "text": "..."},
        {"chunk_id": 1, "page": 1, "text": "..."},
    ]
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        text = page["text"]
        page_num = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size         # where this chunk ends
            chunk_text = text[start:end]     # slice the text

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": chunk_text
            })

            chunk_id += 1
            start = end - overlap            # move forward but keep overlap

    return chunks


# load the model once (outside the function so it doesn't reload every time)
#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
HF_API_KEY = os.getenv("HF_API_KEY")
HF_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def get_embedding(text: str) -> list:
    response = requests.post(
        HF_URL,
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    return response.json()

def embed_chunks(chunks: list) -> list:
    for chunk in chunks:
        chunk["embedding"] = get_embedding(chunk["text"])
    return chunks


chroma_client = chromadb.PersistentClient(path="./chroma_db")

def store_in_chromadb(chunks: list, collection_name: str) -> None:
    """
    Stores chunks and their embeddings in ChromaDB.
    Clears old chunks first so previous PDFs don't mix in.
    Stores chunks in a named collection (one per PDF).
    """
    # delete old collection and create fresh one
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass  # if collection doesn't exist yet, no problem

    collection = chroma_client.get_or_create_collection(name=collection_name)

    ids        = [str(chunk["chunk_id"]) for chunk in chunks]
    embeddings = [chunk["embedding"] for chunk in chunks]
    documents  = [chunk["text"] for chunk in chunks]
    metadatas  = [{"page": chunk["page"]} for chunk in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )


    print(f"Stored {len(chunks)} chunks in collection: {collection_name}")



# ---- Test it ----
if __name__ == "__main__":
    # step 1 - extract
    pages = extract_text_from_pdf("test.pdf")
    print(f"Pages extracted: {len(pages)}")

    # step 2 - chunk
    chunks = split_into_chunks(pages)
    print(f"Chunks created: {len(chunks)}")

    # step 3 - embed
    chunks = embed_chunks(chunks)
    print(f"Chunks embedded: {len(chunks)}")

    # step 4 - store
    store_in_chromadb(chunks)