# ----------------------------------------
# Import tools from LangChain:
# - DirectoryLoader: scans folders for files
# - PyPDFLoader: reads PDF files
# - TextLoader: reads plain text files
# - RecursiveCharacterTextSplitter: splits large text into chunks
# ----------------------------------------
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ----------------------------------------
# STEP 1 — Load documents from disk
# This function:
# - Searches the ./data folder
# - Loads all PDFs with PyPDFLoader
# - Loads all TXT files with TextLoader
# - Combines them into a single list
# ----------------------------------------
def load_documents():
    # Load PDF files
    loader = DirectoryLoader(
        "./Data",           # folder to scan
        glob="**/*.pdf",    # find all PDF files (including subfolders)
        loader_cls=PyPDFLoader
    )
    pdf_docs = loader.load()  # returns a list of Document objects

    # Load text files
    text_loader = DirectoryLoader(
        "./data",
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    text_docs = text_loader.load()  # also returns a list of Document objects

    # Combine PDF + TXT into one list
    return pdf_docs + text_docs


# ----------------------------------------
# STEP 2 — Split documents into chunks
# Why chunks?
# - LLMs can't search huge text directly
# - Chunks (small text pieces) allow:
#   * better embeddings
#   * faster search
#   * accurate answers
#
# chunk_size = max characters per chunk
# chunk_overlap = repeated characters between chunks (helps keep context)
# ----------------------------------------
def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

# ----------------------------------------
# Save chunks into "./chunks" folder
# Each chunk becomes a .txt file
# ----------------------------------------
def save_chunks(chunks, folder="chunks"):
    # Create folder if not exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, chunk in enumerate(chunks):
        filename = f"chunk_{i:04d}.txt"
        path = os.path.join(folder, filename)

        with open(path, "w", encoding="utf-8") as f:
            f.write(chunk.page_content)

    print(f"Saved {len(chunks)} chunks into '{folder}' folder.")

# ----------------------------------------
# MAIN (for testing)
# - Loads all docs
# - Splits them
# - Prints how many were processed
# ----------------------------------------
if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")

    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Generated {len(chunks)} chunks")

    print("Saving chunks...")
    save_chunks(chunks)

    print("Done. You can now run embed.py")