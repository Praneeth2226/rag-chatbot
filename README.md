# 🤖 RAG Chatbot — PDF Question Answering

An AI-powered chatbot that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## ✨ Features

- **PDF Upload** — Upload any PDF document
- **Smart Q&A** — Ask questions and get accurate answers from the document
- **Source References** — See which parts of the document the answer came from
- **Chat History** — Conversational interface with message history
- **No API Key Required** — Uses free HuggingFace models by default

## 🏗️ Architecture

```
PDF Upload → Text Extraction (PyPDF2)
    → Text Chunking (RecursiveCharacterTextSplitter)
    → Embeddings (sentence-transformers/all-MiniLM-L6-v2)
    → Vector Store (FAISS)
    → Query → Retrieve relevant chunks → LLM generates answer
```

## 🛠️ Tech Stack

- **Python** — Core language
- **LangChain** — RAG pipeline orchestration
- **FAISS** — Vector similarity search
- **HuggingFace** — Embeddings & language model
- **PyPDF2** — PDF text extraction
- **Streamlit** — Web UI

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/Praneeth2226/rag-chatbot.git
cd rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📸 Screenshots

*Coming soon*

## 📁 Project Structure

```
rag-chatbot/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # RAG logic (embeddings, retrieval, generation)
├── requirements.txt    # Dependencies
├── .gitignore
└── README.md
```

## 🔧 Configuration

By default, the app uses free HuggingFace models. To use OpenAI instead, set your API key in the Streamlit sidebar.

## 📝 License

MIT

## 👤 Author

**Gamidi Sai Praneeth** — [GitHub](https://github.com/Praneeth2226) | [LinkedIn](https://linkedin.com/in/praneeth-gamidi-965a19265)
