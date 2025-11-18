import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------
# Load FAISS index + metadata
# --------------------------------------------
def load_vector_store(index_path="vector.index", meta_path="meta.pkl"):
    print("Loading FAISS index...")
    index = faiss.read_index(index_path)

    print("Loading metadata (chunk texts)...")
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# --------------------------------------------
# Embed a user question using same model
# --------------------------------------------
def embed_query(question):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embedding = model.encode([question], convert_to_numpy=True)
    return embedding


# --------------------------------------------
# Search FAISS index for similar chunks
# --------------------------------------------
def search(query, top_k=3):
    index, chunks = load_vector_store()

    # Embed the user question
    query_vec = embed_query(query)

    # Search FAISS (returns distances and indices)
    distances, indices = index.search(query_vec, top_k)

    # Collect results
    results = []
    for i, idx in enumerate(indices[0]):
        text = chunks[idx]
        dist = distances[0][i]
        results.append((text, dist))

    return results


# --------------------------------------------
# MAIN (manual test)
# --------------------------------------------
if __name__ == "__main__":
    print("Ask a question:")
    question = input("> ")

    results = search(question, top_k=3)

    print("\n--- Top Results ---\n")
    for i, (text, score) in enumerate(results):
        print(f"[{i+1}] Score: {score}")
        print(text[:500])  # print first 500 chars
        print("\n-------------------\n")