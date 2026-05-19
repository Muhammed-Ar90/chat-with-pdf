import os
os.makedirs("uploads", exist_ok=True)

print("Starting imports...")

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

print("FastAPI imported...")

from ingest import extract_text_from_pdf, split_into_chunks, embed_chunks, store_in_chromadb

print("Ingest imported...")

from query import ask

print("Query imported...")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")


# stores all uploaded PDFs in memory
uploaded_pdfs = {}


class QuestionRequest(BaseModel):
    question: str
    collection_name: str
    chat_history: list = []


#@app.get("/")
#def root():
   # return {"message": "PDF Chat API is running!"}


@app.get("/pdfs")
def get_pdfs():
    """
    Returns list of all uploaded PDFs.
    """
    return {"pdfs": list(uploaded_pdfs.values())}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # save the uploaded file
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # use filename without extension as collection name
    collection_name = file.filename.replace(".pdf", "").replace(" ", "_").lower()

    # run phase 1 pipeline
    pages  = extract_text_from_pdf(file_path)
    chunks = split_into_chunks(pages)
    chunks = embed_chunks(chunks)
    store_in_chromadb(chunks, collection_name)

    # save pdf info
    uploaded_pdfs[collection_name] = {
        "name": file.filename,
        "collection_name": collection_name,
        "pages": len(pages),
        "chunks": len(chunks)
    }

    return {
        "message": "PDF uploaded and processed successfully!",
        "name": file.filename,
        "collection_name": collection_name,
        "pages": len(pages),
        "chunks": len(chunks)
    }


@app.post("/chat")
def chat(request: QuestionRequest):
    result = ask(request.question, request.collection_name, request.chat_history)
    return result


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)