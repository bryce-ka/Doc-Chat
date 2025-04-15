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
from langchain_community.document_loaders import Docx2txtLoader


# Set API keys
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGSMITH_TRACING"] = "True"
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# Initialize model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

st.title("Craft the perfect cold email!")
st.write("Upload your resume along with the job description below")

uploaded_file = st.file_uploader("Upload your Resume", type=["docx"])

def process_doc(file) -> InMemoryVectorStore:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    loader = Docx2txtLoader(tmp_path)
    documents = list(loader.lazy_load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
    all_splits = text_splitter.split_documents(documents)

    embedding_model= OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embedding_model)
    vector_store.add_documents(documents=all_splits)

    return vector_store

@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=5)

# Only process if file is uploaded
if uploaded_file:
    with st.spinner("Processing Resume..."):
        vector_store = process_doc(uploaded_file)

    job_details = st.text_input("Paste the job details below: ")
    
    if job_details:
        with st.spinner("Crafting the message using your background..."):
            result = retriever.invoke(job_details)

            rag_context = " | ".join([doc.page_content for doc in result])

            system_prompt = f"""You are an assistant for writing cold emails to corporate recruiters. 
                                Use the retrived context from the client's resume to write the cold email
                                highlight the client's relevant skills and achievements but keep the message 
                                concise, no more than 2 paragraphs.
                                Additionally give a compatibility score ranging from 1-10 based on how well the client's
                                resume aligns with the job details.
                                Context: {rag_context}:
                            """
            user_message = f"The job details: {job_details}"
            
            st.subheader("Your Cold Email:")
            response_placeholder = st.empty()
            output_text = ""
            for token in model.stream([HumanMessage(content = user_message), SystemMessage(content = system_prompt)]):
                output_text += token.content
                response_placeholder.markdown(output_text)
