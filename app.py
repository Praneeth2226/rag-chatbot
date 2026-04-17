"""
📚 RAG Chatbot — Ask Your PDFs Anything
========================================
A Retrieval-Augmented Generation chatbot that lets you upload PDFs
and have intelligent conversations about their content.

Built with: Streamlit · LangChain · FAISS · HuggingFace
"""

import streamlit as st
from rag_pipeline import RAGPipeline
from ui_components import render_sidebar, render_chat_interface, render_header

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot — Ask Your PDFs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean, modern look */
    .stApp { max-width: 1200px; margin: 0 auto; }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85em;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-ready { background-color: #d4edda; color: #155724; }
    .badge-processing { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "rag_pipeline": None,       # The RAG pipeline instance
        "chat_history": [],         # List of (question, answer, sources) tuples
        "pdfs_processed": False,    # Whether PDFs have been processed
        "pdf_names": [],            # Names of uploaded PDFs
        "llm_provider": "huggingface_api",  # Selected LLM provider
        "hf_api_token": "",         # HuggingFace API token (optional)
        "openai_api_key": "",       # OpenAI API key (optional)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def process_uploaded_pdfs(uploaded_files):
    """Extract text from uploaded PDFs and build the vector store."""
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one PDF file.")
        return

    with st.spinner("🔄 Processing PDFs... This may take a moment."):
        # Initialize the RAG pipeline with selected provider
        pipeline = RAGPipeline(
            llm_provider=st.session_state.llm_provider,
            hf_api_token=st.session_state.hf_api_token or None,
            openai_api_key=st.session_state.openai_api_key or None,
        )

        # Process each uploaded PDF
        all_text = []
        pdf_names = []
        for pdf_file in uploaded_files:
            text_chunks = pipeline.extract_and_chunk_pdf(pdf_file)
            all_text.extend(text_chunks)
            pdf_names.append(pdf_file.name)

        if not all_text:
            st.error("❌ No text could be extracted from the uploaded PDFs.")
            return

        # Build the vector store
        pipeline.build_vector_store(all_text)

        # Save to session state
        st.session_state.rag_pipeline = pipeline
        st.session_state.pdfs_processed = True
        st.session_state.pdf_names = pdf_names
        st.session_state.chat_history = []  # Reset chat on new upload

    st.success(f"✅ Processed {len(pdf_names)} PDF(s) — {len(all_text)} text chunks indexed!")


def ask_question(question: str):
    """Send a question to the RAG pipeline and display the answer."""
    if not st.session_state.rag_pipeline:
        st.warning("⚠️ Please upload and process PDFs first.")
        return

    with st.spinner("🤔 Thinking..."):
        result = st.session_state.rag_pipeline.query(question)

    answer = result["answer"]
    sources = result["sources"]

    # Append to chat history
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources,
    })


# ── Main App Flow ───────────────────────────────────────────────────────────
def main():
    init_session_state()

    # Render the header
    render_header()

    # Render sidebar (PDF upload + settings) — returns uploaded files
    uploaded_files = render_sidebar()

    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if uploaded_files:
            if st.button("🚀 Process PDFs & Build Knowledge Base", use_container_width=True):
                process_uploaded_pdfs(uploaded_files)

    # Show status
    if st.session_state.pdfs_processed:
        st.markdown(
            f'<span class="status-badge badge-ready">✅ Ready — '
            f'{len(st.session_state.pdf_names)} PDF(s) loaded</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Chat interface
    render_chat_interface(ask_question)


if __name__ == "__main__":
    main()
