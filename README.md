# 🤖 DocuMind — RAG Chatbot

**DocuMind** is a document-based **Retrieval-Augmented Generation (RAG) chatbot** built with Python, LangChain, Mistral AI, ChromaDB, and Streamlit.

It allows users to provide documents and ask questions about their content. Instead of relying only on the LLM's existing knowledge, DocuMind retrieves relevant information from the uploaded documents and uses it to generate contextual responses.

## ✨ Features

* 📄 Upload and process documents
* ✂️ Split documents into smaller chunks
* 🔢 Generate embeddings for document chunks
* 🗄️ Store embeddings in a **ChromaDB** vector database
* 🔍 Retrieve relevant document content using semantic search
* 🤖 Generate contextual answers using an **LLM**
* 💬 Interactive chatbot interface built with Streamlit

## 🧠 GenAI Concepts Covered

This project helped me practically understand several fundamental Generative AI concepts:

* **Large Language Models (LLMs)**
* **Prompt Engineering**
* **Text Embeddings**
* **Vector Databases**
* **Semantic Search**
* **Retrieval-Augmented Generation (RAG)**

## ⚙️ How It Works

```text
Document
   ↓
Document Loader
   ↓
Text Splitting / Chunking
   ↓
Embeddings
   ↓
ChromaDB
   ↓
User Question
   ↓
Retriever
   ↓
Relevant Context
   ↓
LLM
   ↓
Generated Response
```

DocuMind follows a typical RAG pipeline where relevant information is retrieved from the document knowledge base before generating the final response.

## 🛠️ Tech Stack

* **Python**
* **LangChain**
* **Mistral AI**
* **ChromaDB**
* **Streamlit**
* **python-dotenv**


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd DocuMind
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file and add your required API key:

```env
MISTRAL_API_KEY=your_api_key_here
```

### 5. Run the Application

```bash
streamlit run main.py 
```

The application will open in your browser.

## 🎯 Learning Outcomes

Through DocuMind, I gained practical experience in:

* Building a complete RAG pipeline
* Working with LLMs and embeddings
* Using vector databases for information retrieval
* Understanding semantic search
* Connecting retrieved context with LLM responses
* Building a simple GenAI application with Streamlit


**Live Link**:   https://documind-rag-chatbot-by-nabeel.streamlit.app/

## 👨‍💻 Author

**Nabeel Shahid**

Computer Science Student | AI/ML & Generative AI

---

⭐ If you find this project useful, consider giving the repository a star!
