import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import chain
from langchain_core.documents import Document
from typing import List
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Set API keys
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGSMITH_TRACING"] = "True"
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# Initialize model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

st.title("📘 Chat with Your PDF")

st.write("Upload a PDF and ask it questions using GPT-4o")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

def process_pdf(file) -> InMemoryVectorStore:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = list(loader.lazy_load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(documents)

    embedding_model= OpenAIEmbeddings(model="text-embedding-3-large")
    # vector_store = InMemoryVectorStore(embedding_model)
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embedding_model,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=all_splits)

    return vector_store

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=3)

# Only process if file is uploaded
if uploaded_file:
    with st.spinner("Processing PDF..."):
        vector_store = process_pdf(uploaded_file)

    user_query = st.text_input("Enter your question:")
    
    if user_query:
        with st.spinner("Retrieving context and generating answer..."):
            result = retriever.invoke(user_query)
            
            rag_context = " | ".join([doc.page_content for doc in result])

            system_prompt = f"""You are an assistant for question-answering tasks. 
                            Use the following pieces of retrieved context to answer the question. 
                            If you don't know the answer, just say that you don't know. 
                            Use three sentences maximum and keep the answer concise.
                            Context: {rag_context}:"""

            st.subheader("Answer:")
            response_placeholder = st.empty()
            output_text = ""
            for token in model.stream([SystemMessage(content = system_prompt), HumanMessage(content = user_query)]):
                output_text += token.content
                response_placeholder.markdown(output_text)

    
