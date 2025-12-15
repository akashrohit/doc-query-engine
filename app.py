import streamlit as st
import os
import tempfile
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PDFPlumberLoader, TextLoader, Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# For local usage, you can paste your token here or set it as an env var
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "PASTE_YOUR_TOKEN_HERE"

@st.cache_resource
def load_local_model():
    """
    Loads LaMini-T5-738M locally.
    This fixes the 404/410 errors by running the brain on your machine.
    """
    model_id = "MBZUAI/LaMini-T5-738M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.3,
        repetition_penalty=1.1
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def process_input(input_type, input_data):
    documents = []

    # 1. Handle Links
    if input_type == "Link":
        # input_data is a list of URLs
        urls = [url.strip() for url in input_data if url.strip()]
        if urls:
            loader = WebBaseLoader(urls)
            documents = loader.load()

    # 2. Handle Text Input
    elif input_type == "Text":
        if input_data.strip():
            documents = [Document(page_content=input_data)]

    # 3. Handle Files (PDF, DOCX, TXT)
    elif input_data is not None:
        # We save the uploaded file to a temp file so LangChain loaders can read it
        # This is CRITICAL for PDFPlumber to fix your spacing issues
        suffix = f".{input_data.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(input_data.read())
            tmp_path = tmp_file.name

        try:
            if input_type == "PDF":
                loader = PDFPlumberLoader(tmp_path)
            elif input_type == "DOCX":
                loader = Docx2txtLoader(tmp_path)
            elif input_type == "TXT":
                loader = TextLoader(tmp_path, encoding="utf-8")
            else:
                st.error("Unsupported file format")
                return None

            documents = loader.load()
        finally:
            # Clean up the temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if not documents:
        return None

    # === FIXED: Better Splitter & Overlap ===
    # This prevents sentences from being cut in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def answer_question(vectorstore, query, llm):
    # === TEACHER PROMPT ===
    # Forces detailed answers and prevents "Yes/No" responses
    template = """You are a helpful academic assistant.

    Context:
    {context}

    Question: {question}

    Task:
    1. Analyze the Context provided above.
    2. Write a detailed paragraph (at least 3 sentences) answering the question.
    3. Include specific definitions or examples from the text if available.

    Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    result = qa_chain.invoke({"query": query})
    return result

def main():
    st.set_page_config(page_title="DocQuery Pro", layout="wide")
    st.title("🤖 DocQuery Pro: High-Accuracy Local RAG")

    with st.spinner("Initializing AI Brain (Local)..."):
        llm = load_local_model()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📂 Data Source")
        input_type = st.selectbox("Input Type", ["Link", "PDF", "Text", "DOCX", "TXT"])
        input_data = None

        if input_type == "Link":
            number_input = st.number_input(min_value=1, max_value=20, step=1, label="Links")
            urls = []
            for i in range(number_input):
                url = st.text_input(f"URL {i+1}")
                if url: urls.append(url)
            input_data = urls
        elif input_type == "Text":
            input_data = st.text_input("Enter text")
        elif input_type == 'PDF':
            input_data = st.file_uploader("Upload PDF", type=["pdf"])
        elif input_type == 'TXT':
            input_data = st.file_uploader("Upload TXT", type=['txt'])
        elif input_type == 'DOCX':
            input_data = st.file_uploader("Upload DOCX", type=['docx', 'doc'])

        if st.button("Build Knowledge Base"):
            if input_data:
                with st.spinner("Reading & Indexing..."):
                    vectorstore = process_input(input_type, input_data)
                    if vectorstore:
                        st.session_state["vectorstore"] = vectorstore
                        st.success("✅ Ready! The AI knows your document.")
            else:
                st.warning("Please provide input.")

    with col2:
        st.header("💬 Chat")
        if "vectorstore" in st.session_state:
            query = st.text_input("Ask a specific question based on the document:")
            if st.button("Get Answer"):
                if query.strip():
                    with st.spinner("Thinking..."):
                        result = answer_question(st.session_state["vectorstore"], query, llm)

                        st.markdown("### 📝 AI Answer:")
                        st.success(result["result"])

                        with st.expander("🕵️ Source Evidence (What the AI read)"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.markdown(f"**Source Chunk {i+1}:**")
                                st.code(doc.page_content)
                else:
                    st.warning("Please enter a question.")
        else:
            st.info("Please build the knowledge base on the left first.")

if __name__ == "__main__":
    main()
