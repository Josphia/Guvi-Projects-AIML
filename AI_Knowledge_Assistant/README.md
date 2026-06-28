# Personal AI Knowledge Assistant 🤖

An advanced Retrieval-Augmented Generation (RAG) application built with LangChain, Streamlit, and Gemini 2.5 Flash that allows users to upload, process, and converse with multi-format enterprise documents natively and privately.

## 🚀 Key Upgrades & Engineering Enhancements
* Parent-Document Retrieval Architecture: Implemented a dual-layer chunking pipeline. The database searches precise 400-character "child" snippets for high semantic fidelity, but returns context-rich 2000-character "parent" blocks to the LLM. This prevents context fragmentation while lowering LLM token consumption overhead.
* Expanded Enterprise Ingestion: Refactored basic text loaders into structured parsers to natively ingest and process `.pdf`, `.docx`, `.csv`, and `.txt` files.
* Source Attribution UI: Enhanced the frontend chat experience by dynamically rendering verified source document references alongside deep contextual text snippets for transparency.
* Production Containerization: Fully containerized via Docker for universal cross-platform deployment stability.

---

## 🛠️ Tech Stack
* Orchestration: LangChain
* LLM Core: Google Gemini 2.5 Flash (`gemini-2.5-flash`)
* Vector Store: FAISS (In-Memory Semantic Index)
* Embeddings: HuggingFace `all-MiniLM-L6-v2`
* Frontend UI: Streamlit

---

## 📦 Local Setup Instructions

### 1. Environment Configuration
Create a `.env` file in the root directory and append your Gemini API key:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here