import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
st.title("📚 Personal AI Knowledge Assistant")
st.subheader("Upload documents and ask questions in natural language.")

import tempfile

def process_documents(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = tmp.name
        
        loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

with st.sidebar:
    st.header("Document Center")
    files = st.file_uploader("Upload PDFs or Text files", accept_multiple_files=True)
    process_btn = st.button("Train Assistant")

    if process_btn and files:
        with st.spinner("Processing documents..."):
            st.session_state.vector_store = process_documents(files)
            st.success("Assistant is ready!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        output_key='answer'
    )

if "vector_store" in st.session_state:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, convert_system_message_to_human=True )
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True 
    )

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            current_avatar = "user.png"
        else:
            current_avatar = "bot.png"
        
        with st.chat_message(message["role"], avatar=current_avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="user.png"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="bot.png"):
            response = qa_chain.invoke({"question": prompt})
            answer = response["answer"]
            
            st.markdown(answer)
            
            with st.expander("View Source References"):
                for doc in response["source_documents"]:
                    st.write(f"- {doc.metadata.get('source', 'Unknown')}")

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload and process documents to start the conversation.")