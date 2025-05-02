import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

def load_file(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    elif file.type == "text/csv":
        df = pd.read_csv(file)
        return df.to_string()
    else:
        return ""


def embed_documents(text_list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(text_list)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def build_chain(vectorstore):
    llm = ChatGroq(model="gemma2-9b-it", temperature=0, api_key="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")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False,
    )
    return chain

st.set_page_config(page_title="RAG Chat App", layout="wide")
st.title("RAG Chat App - Upload Documents & Chat")

# Session state setup
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader("Upload PDF, TXT, or CSV files", type=["pdf", "txt", "csv"], accept_multiple_files=True)
user_question = st.text_input("Ask a question about your documents")

col1, col2 = st.columns([1, 5])
with col1:
    if st.button("Initialize QA"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            all_text = []
            for file in uploaded_files:
                text = load_file(file)
                all_text.append(text)

            vs = embed_documents(all_text)
            st.session_state.chain = build_chain(vs)
            st.session_state.chat_history = []
            st.success("System is ready! Start asking questions.")

with col2:
    if st.session_state.chain and user_question:
        response = st.session_state.chain.run(user_question)
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("AI", response))

# Chat History Display
st.markdown("Chat History")
for speaker, text in st.session_state.chat_history:
    st.markdown(f"{speaker}: {text}")

