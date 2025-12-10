import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough


st.title("📰 Gemini 2.5 Flash RAG — News URL Q&A Bot ")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ------------------------
# INGEST SECTION
# ------------------------
urls_input = st.text_area(
    "Enter news URLs (one per line):",
    placeholder="https://news.com/article1\nhttps://news.com/article2",
)

if st.button("Ingest URLs"):
    urls = [u.strip() for u in urls_input.split("\n") if u.strip()]

    if not urls:
        st.error("Please enter at least one URL.")
        st.stop()

    st.info("📥 Loading URLs...")
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    st.success(f"Loaded {len(docs)} documents.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    st.write(f"📦 Generated {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("🎉 Ingestion complete!")


# ------------------------
# RAG PIPELINE
# ------------------------

question = st.text_input("Ask a question about the ingested news:")

if st.button("Get Answer"):
    if st.session_state.vectorstore is None:
        st.error("Please ingest URLs first.")
        st.stop()

    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    # --- RAG PROMPT ---
    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant answering questions using ONLY the context below.

<context>
{context}
</context>

Question: {question}

Answer clearly and concisely.
""")

    # Build LCEL RAG Chain
    rag_chain = (
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
    )

    with st.spinner("💡 Thinking with Gemini Flash 2.5..."):
        answer = rag_chain.invoke(question)

    st.subheader("💬 Answer")
    st.write(answer.content)

    with st.expander("📚 Retrieved Sources"):
        docs = retriever.invoke(question)
        for i, d in enumerate(docs, 1):
            st.markdown(f"**Source {i}:** {d.metadata.get('source','')}")
            st.code(d.page_content[:400] + "...")

