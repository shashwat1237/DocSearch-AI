import streamlit as st
import tempfile
import os


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


from langchain_groq import ChatGroq



GROQ_API_KEY = #Hidden


st.set_page_config(
    page_title="AI RAG Assistant",
    layout="wide"
)


st.title("AI Document Intelligence (RAG)")
st.caption("Upload multiple PDFs and ask intelligent questions")


with st.sidebar:
    st.header("Control Panel")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")
    st.subheader("System Status")

    if not GROQ_API_KEY:
        st.error("Missing GROQ_API_KEY")
    elif uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) uploaded")
    else:
        st.warning("Awaiting upload...")


def process_pdfs(uploaded_files):
    all_chunks = []

    try:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                path = tmp.name

            loader = PyPDFLoader(path)
            documents = loader.load()

            os.unlink(path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)

        return all_chunks

    except Exception as e:
        st.error(f"PDF Processing Error: {e}")
        return None


def create_rag_chain(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.from_documents(chunks, embeddings)

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": 6}),
            return_source_documents=True
        )

        return qa_chain, len(chunks)

    except Exception as e:
        st.error(f"RAG Setup Error: {e}")
        return None, 0



if uploaded_files and GROQ_API_KEY:

    
    current_files = sorted([f.name for f in uploaded_files])

    if "file_list" not in st.session_state or st.session_state.file_list != current_files:
        st.session_state.clear()
        st.session_state.file_list = current_files

    if "qa_chain" not in st.session_state:

        with st.spinner("Processing documents..."):
            chunks = process_pdfs(uploaded_files)

        if chunks is None:
            st.stop()

        with st.spinner("Creating embeddings..."):
            qa_chain, chunk_count = create_rag_chain(chunks)

        if qa_chain is None:
            st.stop()

        st.session_state.qa_chain = qa_chain
        st.session_state.chunk_count = chunk_count

    qa_chain = st.session_state.qa_chain
    chunk_count = st.session_state.chunk_count

    st.success("System Ready!")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Document Insights")
        st.metric("Documents", len(uploaded_files))
        st.metric("Chunks", chunk_count)
        st.metric("Embedding", "MiniLM")
        st.metric("LLM", "LLaMA 3.3 70B (Groq)")

        with st.expander("Uploaded Files"):
            for f in uploaded_files:
                st.write(f.name)

    with col2:
        st.subheader("Ask Questions")

        query = st.text_input("Enter your question")

        if query:
            try:
                with st.spinner("Thinking..."):
                    result = qa_chain.invoke({"query": query})

                st.write("### Answer")
                st.write(result["result"])

                with st.expander(" Sources"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.write(f"**Source {i+1}:**")
                        st.write(doc.page_content[:300] + "...")

            except Exception as e:
                st.error(f"Query Error: {e}")

else:
    st.info("Upload PDFs and ensure API key is set")
