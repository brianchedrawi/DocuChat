import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --------------------------------------------
# STEP 1 — Load chunk text files from ./chunks
# --------------------------------------------
def load_chunks(folder="chunks"):
    chunks = []
    file_names = []

    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            chunks.append(text)
            file_names.append(filename)

    return chunks, file_names


# --------------------------------------------
# STEP 2 — Create embeddings using a LOCAL model
# Model: all-MiniLM-L6-v2 (free, fast, good)
# --------------------------------------------
def generate_embeddings(chunks):
    print("Loading local embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding chunks (this may take a few seconds)...")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    return embeddings


# --------------------------------------------
# STEP 3 — Save FAISS vector index + metadata
# --------------------------------------------
def save_faiss_index(embeddings, chunks, index_path="vector.index", meta_path="meta.pkl"):
    dim = embeddings.shape[1]  # number of embedding dimensions

    # Build FAISS index with L2 similarity
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_path)

    # Save metadata (the raw chunk texts)
    with open(meta_path, "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index + metadata saved.")


# --------------------------------------------
# MAIN execution
# --------------------------------------------
if __name__ == "__main__":
    print("Loading chunk files...")
    chunks, names = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)

    print("Saving FAISS vector store...")
    save_faiss_index(embeddings, chunks)

    print("Done! Your local vector database is ready.")