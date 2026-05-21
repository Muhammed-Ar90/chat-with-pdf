# Chat with PDF 💬

An AI-powered document assistant that lets you have a natural conversation with any PDF file. Upload a document, ask questions, and get accurate answers with source page citations — instantly.

🔗 **Live Demo:** [chat-with-pdf-doa4.onrender.com](https://chat-with-pdf-doa4.onrender.com)

---

## Features

- 📄 **Upload any PDF** — extract and process documents instantly
- 💬 **Natural language Q&A** — ask questions in plain English
- 🔍 **RAG pipeline** — retrieves the most relevant chunks before answering
- 📌 **Source citations** — every answer shows which page it came from
- 🗂️ **Multi-PDF support** — upload multiple PDFs with separate chat histories
- ⚡ **Fast answers** — powered by LLaMA 3.1 via Groq

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Vector DB | ChromaDB |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| LLM | LLaMA 3.1 8B via Groq |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Render |

---

## How It Works

1. **Upload** — PDF is parsed and split into overlapping text chunks
2. **Embed** — each chunk is converted to a vector embedding via HuggingFace
3. **Store** — embeddings are stored in ChromaDB (one collection per PDF)
4. **Query** — user question is embedded and matched against stored chunks
5. **Answer** — top matching chunks are sent to LLaMA 3.1 as context, which generates the answer

---

## Local Setup

### Prerequisites
- Python 3.11+
- Groq API key — [console.groq.com](https://console.groq.com)
- HuggingFace API key — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Installation

```bash
# Clone the repo
git clone https://github.com/Muhammed-Ar90/chat-with-pdf.git
cd chat-with-pdf

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
HF_API_KEY=your_huggingface_api_key_here
```

### Run

```bash
python main.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Project Structure

```
chat-with-pdf/
├── static/
│   ├── index.html       # Frontend UI
│   └── logo.svg         # App logo
├── uploads/             # Uploaded PDFs (auto-created)
├── chroma_db/           # Vector database (auto-created)
├── main.py              # FastAPI app & routes
├── ingest.py            # PDF parsing, chunking & embedding
├── query.py             # Retrieval & answer generation
├── requirements.txt
├── Procfile
└── .env                 # API keys (not committed)
```

---

## Deployment

Deployed on **Render** as a web service.

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python main.py`
- **Environment Variables:** `GROQ_API_KEY`, `HF_API_KEY`

---

## Author

Built by **Muhammed Abdul Razak**

---

> ⚠️ Note: The free tier on Render has an ephemeral filesystem — uploaded PDFs and ChromaDB data reset on each redeploy. For persistent storage, a paid Render disk or cloud database is recommended.
