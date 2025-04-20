# 🧠 RAGnosis

**RAGnosis** is a Retrieval-Augmented Generation (RAG) system that enhances the contextual understanding of Large Language Models (LLMs) by integrating **external medical knowledge** from the MIMIC dataset. It uses advanced embedding techniques, semantic search, and a generative model to answer clinical queries more accurately.

---

## ✅ Features

- 💡 Uses **Google Generative AI Embeddings** to represent text semantically.
- 🔎 Retrieves top-**k** relevant document chunks using **FAISS**.
- 🤖 Generates precise, context-based responses using **Gemini 1.5 Flash**.
- ⚙️ Chunking size, overlap, and top-**k** retrieval are fully configurable.
- 🖥️ Built with an interactive UI using **Streamlit**.

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/RAGnosis.git
cd RAGnosis
```

### 2. Visit Google Studio AI and create your API key
setup your key in dotenv file.
```bash
GEMINI_API_KEY=your_api_key_here
```
### 3. Install required Libraries
```bash
pip install -r requirements.txt
```
### 3. run the following command
```bash
streamlit run app.py
