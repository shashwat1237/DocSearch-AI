# AI Document Intelligence (RAG)

A Streamlit-based Retrieval-Augmented Generation (RAG) application that lets you upload multiple PDF documents and ask natural language questions against their content. It uses HuggingFace embeddings for semantic search, FAISS for vector storage, and Groq's LLaMA 3.3 70B model for generating accurate, context-aware answers.

---

## What It Does

1. You upload one or more PDF files via the sidebar.
2. The app extracts and chunks the text from each PDF.
3. Each chunk is embedded using a lightweight sentence transformer model (`all-MiniLM-L6-v2`).
4. Embeddings are stored in a FAISS vector index for fast similarity search.
5. When you ask a question, the top 6 most relevant chunks are retrieved and passed to LLaMA 3.3 70B (via Groq API) to generate a grounded answer.
6. Source excerpts are shown alongside the answer so you can verify the context.

---

## Tech Stack

| Component | Library / Model |
|---|---|
| UI Framework | Streamlit |
| PDF Parsing | LangChain `PyPDFLoader` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| LLM | Groq `llama-3.3-70b-versatile` |
| RAG Chain | LangChain `RetrievalQA` |

---

## Project Structure

```
.
├── app.py          # Main Streamlit application
├── README.md       # This file
└── requirements.txt
```

---

## Prerequisites

- Python 3.9+
- A valid [Groq API key](https://console.groq.com/)

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/ai-rag-assistant.git
cd ai-rag-assistant

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
streamlit
langchain
langchain-community
langchain-groq
faiss-cpu
sentence-transformers
pypdf
```

---

## Configuration

The Groq API key is currently hardcoded in the script. For production use, replace it with an environment variable:

```python
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

Then set it in your shell before running:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

---

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## How to Use

1. Open the app in your browser.
2. In the left sidebar, click "Upload PDFs" and select one or more PDF files.
3. Wait for the "Processing documents..." and "Creating embeddings..." spinners to complete.
4. Once "System Ready!" appears, type your question in the text input on the right.
5. The answer will appear below, along with expandable source excerpts showing which parts of the documents were used.

---

## Key Implementation Details

### PDF Processing

Each uploaded PDF is written to a temporary file, loaded with `PyPDFLoader`, then deleted from disk. The text is split into chunks of 1000 characters with a 200-character overlap to preserve context across chunk boundaries.

### Embedding & Retrieval

`all-MiniLM-L6-v2` is a compact but effective sentence transformer that runs locally without any API calls. Embeddings are stored in FAISS, and at query time the top 6 most semantically similar chunks are retrieved (`k=6`).

### Session State Management

The app tracks the list of currently uploaded filenames in `st.session_state`. If the file list changes, the session is cleared and the pipeline is rebuilt from scratch, ensuring stale embeddings are never used.

### LLM Settings

The Groq LLaMA model is configured with `temperature=0.3` to keep answers factual and consistent rather than creative.

---

## Limitations

- Large PDFs with many pages will increase processing and embedding time.
- The FAISS index is in-memory only — it is not persisted between sessions.
- The hardcoded API key should be moved to an environment variable before sharing or deploying.
- Scanned PDFs (image-based) are not supported; the loader requires text-layer PDFs.

---

## License

MIT
