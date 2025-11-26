# Full corrected app.py

import streamlit as st
import os
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from openai import OpenAI

# ==========================================================
# Streamlit Chat Interface for RAG Chatbot
# + Dynamic File Upload for embeddings
# + Conversation mode
# + Proper citations
# ==========================================================

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "docs"
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

OPENROUTER_KEY = st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)

LLM_MODEL = "google/gemma-3-12b-it:free"


# ----------------------------------------------------------
# INITIALIZATION (Cached)
# ----------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    return collection


embedder = load_embedding_model()
collection = load_chroma()


# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------
def embed_text(text: str):
    emb = embedder.encode([text], convert_to_numpy=True)[0]
    return emb.tolist()


def retrieve_context(query: str, k=5):
    query_vector = embed_text(query)

    results = collection.query(
        query_embeddings=[query_vector], n_results=k, include=["documents", "metadatas"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return list(zip(docs, metas))


def build_conversation_history():
    """
    Convert chat history into a clean string for LLM memory.
    Only keeps the last few messages for efficiency.
    """
    history = st.session_state.get("messages", [])

    # keep last 6 messages total (3 user + 3 assistant)
    trimmed = history[-6:]

    formatted = ""
    for msg in trimmed:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted += f"{role}: {msg['content']}\n"

    return formatted


def ask_llm(query: str, context: str) -> str:
    """Send conversation + context + question to OpenRouter."""

    # Get conversation history from session state
    history = st.session_state.get("history", [])

    # Format conversation history as text (last 6 messages)
    conversation = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n"

    # Build the complete prompt
    prompt = f"""
You are a helpful, friendly, conversational assistant.

You have two sources of information:
1. **Conversation memory** (general conversation context)
2. **Document context** (may or may not be relevant to the question)

Your rules:
- Use conversation memory to stay natural and maintain flow
- If the user's question is about the documents → use the document context
- If the documents are not relevant → answer using general knowledge
- NEVER hallucinate document facts
- NEVER mention the document context unless asked for a document-specific answer

CONVERSATION SO FAR:
{conversation}

DOCUMENT CONTEXT (may or may not be relevant):
{context}

USER QUESTION:
{query}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.set_page_config(page_title="DocChat", page_icon="📄", layout="centered")
st.title("📄 DocChat — Internal Document Assistant")

# ==========================================================
# FILE UPLOAD
# ==========================================================
st.subheader("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("uploaded_data", exist_ok=True)

    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    for file in uploaded_files:
        file_path = os.path.join("uploaded_data", file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        st.success(f"📥 Saved {file.name}")

        loader = (
            PyPDFLoader(file_path)
            if file.name.endswith(".pdf")
            else TextLoader(file_path)
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        st.info(f"🔹 Created {len(chunks)} chunks from {file.name}")

        added = 0
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file.name}-{i}"

            existing = collection.get(ids=[chunk_id])
            if existing["ids"]:
                continue

            embedding = embedder.encode(chunk.page_content).tolist()

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{"source_file": file.name}],
                documents=[chunk.page_content],
            )
            added += 1

        st.success(f"✅ Added {added} chunks from {file.name}")

st.divider()

# ==========================================================
# CHAT
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content}

if "history" not in st.session_state:
    st.session_state.history = []  # LLM conversation history

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_query = st.chat_input("Ask anything about your documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = retrieve_context(user_query, k=5)
            combined_context = "\n\n".join([txt for txt, meta in results])

            answer = ask_llm(user_query, combined_context)

            # Determine whether the answer used the document context
            used_docs = (
                "from the provided documents" in answer.lower()
                or "according to the document" in answer.lower()
                or any(txt[:100].lower() in answer.lower() for txt, _ in results)
            )

            doc_tag = "[USED_DOCS: yes]" if used_docs else "[USED_DOCS: no]"
            answer_with_tag = f"{answer}\n\n{doc_tag}"

            st.write(answer)

            # Store ONLY the clean answer in messages (for display)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Store answer WITH tag in a separate tracking list (optional, for analytics)
            # st.session_state.messages.append({"role": "assistant", "content": answer_with_tag})

            # Store ONLY the clean answer in history (sent to LLM)
            st.session_state.history.append({"role": "assistant", "content": answer})

            # Only show sources if documents were used
            if used_docs:
                st.markdown("### 📚 Sources used:")
                for i, (txt, meta) in enumerate(results):
                    st.markdown(f"**{i+1}. {meta.get('source_file', 'Unknown File')}**")
                    st.code(txt[:200] + ("..." if len(txt) > 200 else ""))
# END
