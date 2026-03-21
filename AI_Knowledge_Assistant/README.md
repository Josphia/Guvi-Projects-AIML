# 🤖 AI Knowledge Assistant

A smart chatbot that allows users to upload documents and ask questions, receiving accurate answers based on the document content.

---

## 🧠 How It Works

1. User uploads documents
2. Documents are split into smaller chunks
3. Text is converted into embeddings
4. FAISS stores embeddings for fast retrieval
5. User asks a question
6. Relevant chunks are retrieved
7. Gemini generates an answer based on context

---

## 🧠 Built With

Frontend: Streamlit
Backend: Python
LLM: Google Gemini (gemini-2.5-flash)
Embeddings: HuggingFace (all-MiniLM-L6-v2)
Vector Store: FAISS
Framework: LangChain

---

## ▶️ Run

```bash
streamlit run Main_Page.py
```

---

## 💡 About

This project uses **RAG (Retrieval-Augmented Generation)** to understand and answer questions from documents.

---