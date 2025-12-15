# DocQuery Engine: High-Accuracy Local RAG

**DocQuery Engine** is a secure, offline-capable Retrieval-Augmented Generation (RAG) application. It enables users to upload documents (PDF, DOCX, TXT) or provide Web Links and ask questions about them using a local Large Language Model (LLM).

Unlike standard RAG apps that rely on expensive external APIs, this project uses **LaMini-T5** running locally on your machine, ensuring **data privacy**, **zero latency**, and **zero cost**.

## 🚀 Key Features

* **100% Local Inference:** Runs the `MBZUAI/LaMini-T5-738M` model locally using Hugging Face pipelines. No OpenAI API keys required.
* **Multi-Format Support:** Seamlessly ingests **PDFs**, **Word Documents (.docx)**, **Text files**, and **Web URLs**.
* **Smart Text Cleaning:** Includes a custom preprocessing engine that repairs "mashed" text in PDFs (e.g., fixing `24Abstraction` to `24 Abstraction`) using `pdfplumber` and Regex.
* **High-Fidelity Retrieval:** Utilizes `RecursiveCharacterTextSplitter` and `FAISS` vector stores to maintain context across document chunks.
* **"Teacher Mode" Answers:** Custom prompt templates force the AI to provide detailed, 3-4 sentence explanations rather than simple "Yes/No" answers.

## 🛠️ Tech Stack

* **Framework:** Streamlit
* **Orchestration:** LangChain
* **LLM:** Hugging Face Transformers (`LaMini-T5-738M`)
* **Vector Database:** FAISS (CPU)
* **Embeddings:** Sentence-Transformers (`all-mpnet-base-v2`)
* **Document Parsers:** `pdfplumber`, `python-docx`, `BeautifulSoup4`
