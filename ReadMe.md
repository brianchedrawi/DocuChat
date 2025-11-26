# 📄 DocChat - Internal Document Assistant

A conversational RAG (Retrieval-Augmented Generation) chatbot that allows you to upload documents and ask questions about them. Built with Streamlit, ChromaDB, and powered by Google's Gemma model via OpenRouter.

## ✨ Features

- 📤 **Upload PDF/TXT files** and automatically embed them
- 💬 **Conversational interface** with memory across the session
- 🔍 **Smart retrieval** using semantic search
- 📚 **Source citations** showing which documents were used
- 🧠 **Hybrid responses** - uses documents when relevant, general knowledge otherwise

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenRouter API key ([Get one here](https://openrouter.ai/))

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up secrets**

Create a `.streamlit/secrets.toml` file in the project root:
```toml
OPENROUTER_API_KEY = "your-api-key-here"
```
## 📖 Usage

DocChat supports **two methods** for adding documents:

### Method 1: UI Upload (Quick & Interactive) ⚡
**Best for:** Testing, small batches, demos

1. Open the app in your browser
2. Click **"Browse files"** and upload PDF or TXT files
3. Files are automatically processed and stored
4. Start chatting immediately

### Method 2: Batch Ingestion (Efficient & Scalable) 🚀
**Best for:** Initial bulk load, large datasets, automation

1. Place your PDF/TXT files in the `./data` folder
2. Run the ingestion script:
```bash
   python ingest.py
```
3. Files are processed with:
   - ✅ SHA256 hash tracking (no duplicates)
   - ✅ Incremental processing (only new files)
   - ✅ Manifest tracking in `chroma_store/manifest.json`
4. Launch the app:
```bash
   streamlit run app.py
```

**Both methods** store embeddings in the same ChromaDB collection, so you can mix and match!

4. **Run the app**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Upload Documents**: Click "Browse files" and upload your PDF or TXT files
2. **Ask Questions**: Type your question in the chat input
3. **View Sources**: When the answer uses your documents, sources are displayed below

## 🏗️ Architecture
```
User Query
    ↓
Semantic Search (ChromaDB)
    ↓
Retrieve Top-K Documents
    ↓
LLM (Gemma 3) with Context
    ↓
Response + Citations
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: SentenceTransformers (all-mpnet-base-v2)
- **LLM**: Google Gemma 3 via OpenRouter
- **Document Processing**: LangChain, PyPDF

## 📁 Project Structure
```
.
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── secrets.toml      # API keys (DO NOT COMMIT)
├── chroma_store/         # Vector database (auto-generated)
└── uploaded_data/        # Uploaded files (auto-generated)
```

## 🔒 Security Notes

- Never commit your `.streamlit/secrets.toml` file
- Add it to `.gitignore`
- For production, use environment variables or Streamlit Cloud secrets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - feel free to use this project for your own purposes.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [OpenRouter](https://openrouter.ai/)
- Embeddings by [SentenceTransformers](https://www.sbert.net/)

---

Made with ❤️ by [Brian Chedrawi]