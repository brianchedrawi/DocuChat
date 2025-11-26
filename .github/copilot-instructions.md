# DocChat AI Coding Agent Instructions

## Project Overview

**DocChat** is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and ask questions about them. The system uses local embeddings (Sentence Transformers) and vector storage (ChromaDB) paired with an LLM (OpenRouter API) to generate context-aware answers.

### Key Architecture Components

1. **Document Ingestion** (`ingest.py`): Processes PDF/TXT files, chunks text, generates embeddings, stores in ChromaDB with SHA256-based deduplication
2. **Vector Storage** (`chroma_store/`): Persistent ChromaDB instance storing document embeddings and chunks
3. **Chat Interface** (`app.py`): Streamlit web UI for uploading documents and querying with retrieval-augmented context
4. **Query Pipeline** (`query.py`): Standalone RAG script demonstrating retrieve-then-LLM pattern

### Technology Stack

- **Vector Database**: ChromaDB (persistent, local)
- **Embeddings**: Sentence Transformers (`all-mpnet-base-v2`)
- **LLM**: OpenRouter API (`google/gemma-3-27b-it:free`)
- **UI Framework**: Streamlit
- **Text Splitting**: LangChain's `RecursiveCharacterTextSplitter`
- **PDF Parsing**: PyPDF/LangChain loaders
- **Alternative Vector Engine**: FAISS (in `Local Retriever with FAISS/` for reference)

---

## Critical Data Flows

### 1. Document Ingestion → Embedding Storage

```
PDF/TXT files in ./data
    ↓ [ingest.py]
    ├─ SHA256 hash computed (deduplication key)
    ├─ Text extracted & chunked (200 chars, 100 overlap)
    ├─ Embeddings generated (all-mpnet-base-v2)
    └─ Stored in ChromaDB with metadata (source file, chunk index)
         └─ manifest.json tracks processed files to skip re-processing
```

**Key Pattern**: Deduplication via `manifest.json` contains file hashes → prevents re-embedding identical files.

### 2. Query → Retrieval → LLM Answer

```
User question in Streamlit UI (app.py)
    ↓ [embed_text()]
    Embed query using same model as ingestion
    ↓ [retrieve_context(query, k=5)]
    ChromaDB finds top-5 most similar chunks (cosine similarity)
    ↓ [ask_llm(query, context)]
    Combine retrieved chunks + query into prompt
    └─ Send to OpenRouter with Gemma 3.27B model
         └─ Return answer to UI
```

**Key Pattern**: Embeddings use **same model** for both ingestion and retrieval (critical for consistency).

### 3. Dynamic Upload in Streamlit

`app.py` handles real-time file upload without requiring restart:
- Files saved to `./uploaded_data`
- Chunks generated on-the-fly
- Embeddings added directly to live ChromaDB collection
- Duplicate detection via collection IDs (`{filename}-{chunk_index}`)

---

## Code Organization & Patterns

### Module Responsibilities

| File | Purpose | When to Edit |
|------|---------|---|
| `ingest.py` | Batch document processing with deduplication | Adding new file types, changing chunk strategy |
| `app.py` | Streamlit UI + live query interface | UI changes, prompt engineering, retrieval params |
| `query.py` | Standalone RAG demo (CLI) | Testing LLM behavior, debugging retrieve-then-answer |
| `store.py` | (Unused) Old embedding storage reference | Deprecated—ignore |
| `process_docs.py` | (Partial) Document loading patterns | Reference only—use `ingest.py` instead |

### Critical Configuration Points

```python
# app.py & ingest.py
CHROMA_DIR = "./chroma_store"           # Persistent vector DB location
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Must match across all files
CHUNK_SIZE = 200                        # Characters per chunk (tweak for your domain)
CHUNK_OVERLAP = 100                     # 50% overlap for continuity
COLLECTION_NAME = "docs"                # ChromaDB collection name

# app.py only
LLM_MODEL = "google/gemma-3-27b-it:free"  # Free-tier model (must match OpenRouter catalog)
OPENROUTER_KEY = st.secrets["OPENROUTER_API_KEY"]  # Sourced from .streamlit/secrets.toml
```

### Embedding Model Choice Rationale

- **all-mpnet-base-v2**: 384 dimensions, good semantic understanding, balances speed/quality
- Must be consistent across `ingest.py`, `app.py`, and `query.py`
- **Do not change model without re-ingesting all documents**

---

## Developer Workflows

### Running the Project

```bash
# 1. Install dependencies
pip install -r requirement.txt

# 2. Ingest documents (batch processing with deduplication)
python ingest.py
# → Scans ./data, hashes files, embeds chunks, stores in ChromaDB, updates manifest.json

# 3. Launch Streamlit UI (interactive chat + live upload)
streamlit run app.py
# → Visit http://localhost:8501
# → Upload PDFs/TXT in UI to add to collection without restarting

# 4. Query via CLI (standalone testing)
python query.py
# → Interactive prompt for debugging retrieval/LLM behavior
```

### Local Development Testing

- **Test retrieval quality**: Use `query.py` to experiment with `top_k` parameter and prompt wording
- **Debug embeddings**: Check ChromaDB metadata for chunk provenance via:
  ```python
  import chromadb
  client = chromadb.PersistentClient(path="./chroma_store")
  collection = client.get_collection("docs")
  collection.get(limit=5)  # Inspect stored chunks & metadata
  ```
- **Verify deduplication**: Check `chroma_store/manifest.json` for processed file hashes

### Secrets Management

- `.streamlit/secrets.toml` contains `OPENROUTER_API_KEY` (not in repo)
- Required for `app.py` to function; will crash on missing key
- `query.py` has hardcoded key for standalone testing (security risk—replace before production)

---

## Common Patterns & Conventions

### 1. Text Chunking Strategy

Always use `RecursiveCharacterTextSplitter` with:
- `chunk_size=200`: Short enough for precise retrieval, long enough for context
- `chunk_overlap=100`: 50% overlap prevents answer splitting across chunk boundaries

**Why recursive**: Splits on natural boundaries (newlines → sentences → words) instead of arbitrary character breaks.

### 2. Embedding Pipeline Pattern

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Batch embed for efficiency (avoid per-item encoding)
embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

# Convert numpy → lists for ChromaDB
embeddings_list = [emb.tolist() for emb in embeddings]
```

**Key detail**: `convert_to_numpy=True` returns numpy arrays (faster); ChromaDB requires lists (via `.tolist()`).

### 3. ChromaDB Collection Schema

```python
collection.add(
    ids=[f"{file_hash}_{chunk_idx:04d}"],  # Unique ID pattern: file_hash_chunk_number
    documents=[chunk_text],                 # Raw text for display
    embeddings=[embedding_list],            # 384-dim vector
    metadatas=[{"source_file": name, "file_hash": hash, "chunk_index": idx}]  # Provenance
)
```

**Pattern**: Metadata enables tracing retrieved chunks back to source file and location.

### 4. Prompt Engineering Convention

All LLM calls use this structure (see `query.py` and `app.py`):

```
You are an assistant answering based ONLY on the provided context.

CONTEXT:
[Retrieved chunks here]

QUESTION:
[User query here]

If answer not found in context, reply: "I don't have enough information..."
```

**Why explicit instruction**: Prevents LLM from hallucinating outside context; critical for RAG reliability.

### 5. File Processing Pipeline (ingest.py)

```
1. Compute SHA256 hash of raw bytes → uniqueness key
2. Load manifest.json to check if file already processed
3. SKIP if hash in manifest (deduplication)
4. Else: Extract text → Chunk → Embed → Store → Update manifest
```

**Why SHA256 on bytes, not filename**: Handles renamed/moved files and detects actual content changes.

---

## Integration Points & Dependencies

### External APIs

- **OpenRouter**: Provides LLM via HTTP. Requires `OPENROUTER_KEY`, `HTTP-Referer` header, and `X-Title` header (see `query.py` extra_headers)
- **Sentence Transformers Hub**: Downloads model automatically on first run (cached locally)

### File System Dependencies

- `./data/`: Place PDFs/TXT here before running `ingest.py`
- `./chroma_store/`: Persistent ChromaDB directory (created automatically)
- `./chroma_store/manifest.json`: Tracks processed files (created by `ingest.py`)
- `./uploaded_data/`: Staging folder for Streamlit file uploads (created by `app.py`)
- `.streamlit/secrets.toml`: Configuration file with API keys (local only)

### Model Caching

- Sentence Transformer downloads to `~/.cache/huggingface/` on first import
- FAISS index saved as `vector.index` and metadata as `meta.pkl` (legacy, not currently used)

---

## Known Limitations & Future Considerations

1. **No multi-language support**: Embedding model optimized for English; behavior on other languages untested
2. **No chunking strategies for code**: RCTS treats all text equally; consider domain-specific splitters for technical docs
3. **No query expansion**: Single-turn queries only; multi-turn context history not implemented
4. **Free-tier LLM**: Gemma 3.27B may hallucinate on complex reasoning; consider paid models for production
5. **No error recovery**: Failed API calls not retried; transient failures will crash Streamlit session
6. **Vector index not optimized**: ChromaDB default HNSW config acceptable for small corpora (<100k chunks); consider tuning for scale

---

## AI Agent Quick Reference

When modifying this codebase:

- **Changing embedding model**: Update `EMBED_MODEL_NAME` in all files AND re-ingest all documents
- **Adding new file types**: Extend `load_text_from_file()` in `ingest.py` with new loader
- **Improving retrieval quality**: Tune `CHUNK_SIZE`, `CHUNK_OVERLAP`, `top_k` in retrieval calls, or test different `all-mpnet` variants
- **Changing LLM**: Update `LLM_MODEL` in `app.py` and `query.py`; test prompt compatibility
- **Scaling to millions of chunks**: Migrate to FAISS (see `Local Retriever with FAISS/`) or evaluate production ChromaDB tuning
