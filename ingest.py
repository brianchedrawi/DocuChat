#!/usr/bin/env python3
"""
ingest.py

Ingest pipeline (single script):
- Scans ./data for PDF and TXT files
- Computes SHA256 of each file (raw bytes) as document fingerprint
- Skips files already processed (based on manifest.json)
- Loads text (PyPDFLoader for PDF, plain read for TXT)
- Splits text into chunks with overlap
- Generates local embeddings with sentence-transformers/all-mpnet-base-v2
- Saves vectors + metadata to ChromaDB (./chroma_store)
- Updates manifest.json with processed files

Run:
    python ingest.py
"""

import os
import json
import hashlib
from pathlib import Path
import sys
import time
import traceback
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

# Silence tokenizers parallelism warning early
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# External libs (ensure installed)
# pip install sentence-transformers chromadb langchain pypdf
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb

# (Optional) if you want to use langchain loaders for PDF:
try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False

# ---------- CONFIG ----------
DATA_DIR = Path("./data")
CHROMA_DIR = Path("./chroma_store")
MANIFEST_PATH = CHROMA_DIR / "manifest.json"   # tracks processed file hashes
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 200         # characters per chunk (tweakable)
CHUNK_OVERLAP = 100      # overlap between chunks
COLLECTION_NAME = "docs"
# ----------------------------

def compute_file_hash(path: Path) -> str:
    """Return SHA256 hex digest of file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_text_from_file(path: Path) -> str:
    """Load text for PDF or TXT. For PDF we use PyPDFLoader if available."""
    if path.suffix.lower() == ".pdf":
        if PDF_SUPPORT:
            loader = PyPDFLoader(str(path))
            pages = loader.load_and_split()  # returns list of Documents (LangChain)
            # join page contents
            texts = [p.page_content for p in pages]
            return "\n\n".join(texts)
        else:
            # fallback: try very simple PDF read via pypdf (if available)
            try:
                import pypdf
                reader = pypdf.PdfReader(str(path))
                out = []
                for p in reader.pages:
                    out.append(p.extract_text() or "")
                return "\n\n".join(out)
            except Exception:
                raise RuntimeError("PDF support missing. Install langchain and pypdf.")
    else:
        # plain text
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_text(text: str, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Return list of chunk strings using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # splitter.split_text expects a string and returns list of strings
    return splitter.split_text(text)


def load_manifest(manifest_path: Path):
    """Load manifest mapping file_hash -> metadata."""
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(manifest: dict, manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def ensure_chroma_collection(client, name=COLLECTION_NAME):
    """Get or create collection. Returns collection object."""
    try:
        collection = client.get_collection(name)
    except Exception:
        collection = client.create_collection(name)
    return collection


def main():
    start_time = time.time()
    print("INGEST: Starting ingestion pipeline...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest = load_manifest(MANIFEST_PATH)

    # Initialize Chroma client (persistent)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # get or create collection
    try:
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception:
        collection = ensure_chroma_collection(client, COLLECTION_NAME)

    # Load embedding model (local)
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (this may take a bit)...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Scan data folder
    files = sorted([p for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".txt"}])
    if not files:
        print("No files found in ./data. Drop PDFs or TXT files there and re-run.")
        return

    processed_any = False
    for file_path in files:
        try:
            file_hash = compute_file_hash(file_path)
            existing = manifest.get(file_hash)
            if existing:
                print(f"SKIP: {file_path.name} (already processed at {existing.get('processed_at')})")
                continue

            print(f"PROCESSING: {file_path.name}")

            # load text
            text = load_text_from_file(file_path)
            if not text or len(text.strip()) == 0:
                print(f"  WARNING: no text extracted from {file_path.name}, skipping.")
                continue

            # chunk text
            chunks = chunk_text(text)
            print(f"  -> generated {len(chunks)} chunks")

            # embed chunks in batches (to avoid memory bloat)
            batch_size = 64
            embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                embs = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
                embeddings.extend(embs)

            # Prepare ids/metadata
            # use file_hash + index so each embedding has unique id
            ids = [f"{file_hash}_{idx:04d}" for idx in range(len(chunks))]
            metadatas = [{"source_file": file_path.name, "file_hash": file_hash, "chunk_index": idx} for idx in range(len(chunks))]

            # Convert numpy embeddings to lists (Chroma expects python lists)
            embeddings_list = [emb.tolist() for emb in embeddings]

            # Add to Chroma
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings_list,
                metadatas=metadatas
            )

            # Update manifest
            manifest[file_hash] = {
                "file_name": file_path.name,
                "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_chunks": len(chunks)
            }
            save_manifest(manifest, MANIFEST_PATH)
            processed_any = True
            print(f"  -> Saved {len(chunks)} embeddings for {file_path.name}")

        except Exception as e:
            print(f"ERROR processing {file_path.name}: {e}")
            traceback.print_exc()

    elapsed = time.time() - start_time
    if processed_any:
        print(f"\nINGEST: Done. Time elapsed: {elapsed:.1f}s")
    else:
        print("\nINGEST: Nothing new to process. All files are up-to-date.")


if __name__ == "__main__":
    main()