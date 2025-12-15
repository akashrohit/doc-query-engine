# DocQuery Engine: High-Accuracy Local RAG

**DocQuery Engine** is a secure, offline-capable Retrieval-Augmented Generation (RAG) application. It enables users to upload documents (PDF, DOCX, TXT) or provide Web Links and ask questions about them using a local Large Language Model (LLM).

Unlike standard RAG apps that rely on expensive external APIs, this project uses local open-source models, ensuring **data privacy**, **zero latency**, and **zero cost**.

---

## Note on Model Configuration
**Current Configuration:** This repository is pre-configured with **LaMini-T5-738M** to ensure it runs smoothly on the free tier of Streamlit Cloud (CPU-only environments).

**Production/Resume Config:** For the full reasoning capabilities described in my portfolio (context-aware, hallucination-free generation), this system is architected to swap the loader for **Meta Llama 3 (8B)** or **Mistral-7B** via Ollama or Hugging Face pipelines when running on local GPU hardware.

---

## 🚀 Key Features

* **Modular LLM Architecture:** Designed to support lightweight models (LaMini) for cloud demos and high-performance models (Meta Llama 3) for deep reasoning tasks.
* **100% Local Inference:** Runs entirely using Hugging Face pipelines with no dependency on paid OpenAI API keys.
* **Multi-Format Ingestion:** Seamlessly processes **PDFs** (via `pdfplumber`), **Word Docs** (via `docx2txt`), **Text files**, and **Web URLs** (via `WebBaseLoader`).
* **Context-Aware Retrieval:** Utilizes `RecursiveCharacterTextSplitter` to manage context windows and `FAISS` for dense vector similarity search.
* **"Teacher Mode" Prompting:** Custom LangChain prompt templates force the model to provide detailed, evidence-backed answers instead of generic one-liners.

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **Orchestration:** LangChain (Chains & Prompts)
* **LLM (Demo):** LaMini-T5-738M (Optimized for CPU/Cloud)
* **LLM (Production):** Meta Llama 3 / Mistral-7B (Compatible via `AutoModelForCausalLM`)
* **Vector Database:** FAISS (CPU)
* **Embeddings:** Sentence-Transformers (`all-mpnet-base-v2`)
* **Document Parsers:** `pdfplumber`, `docx2txt`, `LangChain WebBaseLoader`
