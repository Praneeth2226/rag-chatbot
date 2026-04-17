"""
RAG Pipeline — Core retrieval-augmented generation logic.
==========================================================
Handles: PDF extraction → text chunking → embedding → vector store → QA chain.

Supports three LLM backends:
  1. HuggingFace Inference API (free tier, needs optional token)
  2. OpenAI API (needs key)
  3. Local pipeline fallback (no API key needed, runs on CPU)
"""

import os
from typing import Optional
from io import BytesIO

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# ── Constants ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_LLM_MODEL = "google/flan-t5-base"

CHUNK_SIZE = 800          # Characters per chunk
CHUNK_OVERLAP = 200       # Overlap between chunks for context continuity
TOP_K_RESULTS = 4         # Number of relevant chunks to retrieve

# Prompt template for the QA chain
QA_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question. 
If you don't know the answer based on the context, say "I don't have enough information in the uploaded documents to answer this question."
Always reference which part of the document your answer comes from.

Context:
{context}

Question: {question}

Helpful Answer:"""


class RAGPipeline:
    """
    End-to-end RAG pipeline: PDF → chunks → embeddings → FAISS → LLM answer.

    Args:
        llm_provider: One of "huggingface_api", "openai", or "local"
        hf_api_token: Optional HuggingFace API token for inference API
        openai_api_key: Optional OpenAI API key
    """

    def __init__(
        self,
        llm_provider: str = "huggingface_api",
        hf_api_token: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.hf_api_token = hf_api_token
        self.openai_api_key = openai_api_key

        # Initialize the embedding model (runs locally, always free)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.vector_store = None
        self.qa_chain = None

    # ── PDF Extraction ──────────────────────────────────────────────────────

    def extract_and_chunk_pdf(self, pdf_file) -> list[Document]:
        """
        Extract text from a PDF file and split it into chunks.

        Args:
            pdf_file: Streamlit UploadedFile object

        Returns:
            List of LangChain Document objects with metadata
        """
        reader = PdfReader(BytesIO(pdf_file.read()))
        documents = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                # Split page text into chunks
                chunks = self.text_splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_file.name,
                            "page": page_num,
                            "chunk": i + 1,
                        },
                    )
                    documents.append(doc)

        # Reset file pointer for potential re-reads
        pdf_file.seek(0)
        return documents

    # ── Vector Store ────────────────────────────────────────────────────────

    def build_vector_store(self, documents: list[Document]):
        """
        Create FAISS vector store from document chunks.

        Args:
            documents: List of LangChain Document objects
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self._build_qa_chain()

    # ── LLM Selection ───────────────────────────────────────────────────────

    def _get_llm(self):
        """
        Initialize the LLM based on the selected provider.

        Returns:
            A LangChain-compatible LLM instance
        """
        if self.llm_provider == "openai" and self.openai_api_key:
            return self._get_openai_llm()
        elif self.llm_provider == "huggingface_api":
            return self._get_hf_api_llm()
        else:
            return self._get_local_llm()

    def _get_hf_api_llm(self):
        """HuggingFace Inference API — free tier with optional token."""
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=HF_LLM_MODEL,
            huggingfacehub_api_token=self.hf_api_token or os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            temperature=0.3,
            max_new_tokens=512,
        )

    def _get_openai_llm(self):
        """OpenAI API — requires API key."""
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=self.openai_api_key,
            temperature=0.3,
            max_tokens=512,
        )

    def _get_local_llm(self):
        """Local pipeline fallback — no API key needed, runs on CPU."""
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

        tokenizer = AutoTokenizer.from_pretrained(HF_LLM_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(HF_LLM_MODEL)

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.3,
        )

        return HuggingFacePipeline(pipeline=pipe)

    # ── QA Chain ────────────────────────────────────────────────────────────

    def _build_qa_chain(self):
        """Build the RetrievalQA chain with the selected LLM."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Process PDFs first.")

        llm = self._get_llm()

        prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": TOP_K_RESULTS},
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    # ── Query ───────────────────────────────────────────────────────────────

    def query(self, question: str) -> dict:
        """
        Ask a question against the indexed PDFs.

        Args:
            question: User's question string

        Returns:
            Dict with 'answer' and 'sources' (list of source metadata dicts)
        """
        if not self.qa_chain:
            return {
                "answer": "⚠️ No documents loaded. Please upload and process PDFs first.",
                "sources": [],
            }

        try:
            result = self.qa_chain.invoke({"query": question})
        except Exception as e:
            return {
                "answer": f"❌ Error generating answer: {str(e)}. Try switching to 'Local (No API)' mode in settings.",
                "sources": [],
            }

        # Extract source information
        sources = []
        seen = set()
        for doc in result.get("source_documents", []):
            meta = doc.metadata
            key = (meta.get("source", ""), meta.get("page", 0))
            if key not in seen:
                sources.append({
                    "file": meta.get("source", "Unknown"),
                    "page": meta.get("page", "?"),
                    "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                })
                seen.add(key)

        return {
            "answer": result.get("result", "No answer generated."),
            "sources": sources,
        }
