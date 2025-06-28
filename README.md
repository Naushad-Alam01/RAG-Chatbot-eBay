# 📄 RAG Chatbot for eBay User Agreement — Amlgo Labs Assignment

## 🚀 Project Overview
This is an AI-powered chatbot capable of answering user queries based on the eBay User Agreement document. It uses a Retrieval-Augmented Generation (RAG) pipeline with a vector database and a language model (LLM).

The chatbot supports real-time streaming responses via a Streamlit interface and retrieves information grounded in the provided document.

---

## 🛠️ Tech Stack
- **Embedding Model:** `all-MiniLM-L6-v2`
- **LLM:** `google/flan-t5-large`
- **Vector Database:** FAISS
- **Frontend:** Streamlit (with streaming response)
- **Programming Language:** Python

---

## ✅ Features
- 🔍 Document-aware, accurate answers
- 🔗 Shows source text chunks used for generating the answer
- 🧠 Memory of chat history during the session
- 🚀 Real-time streaming responses
- 🖥️ Simple and clean web interface

---

## 📁 Folder Structure

RAG-Chatbot-eBay/
├── data/ → Raw document (eBay user agreement)
├── vectordb/ → FAISS index and saved chunks
├── notebooks/ → Jupyter notebook for preprocessing & vector DB creation
├── app.py → Streamlit chatbot app
├── requirements.txt → Python dependencies
└── README.md → Project documentation


---

## 🔧 Installation & Running

### ✅ 1. Clone the repository:

```bash
git clone https://github.com/your-username/RAG-Chatbot-eBay.git
cd RAG-Chatbot-eBay


### ✅ 2. Create virtual environment:

python -m venv ragenv
.\ragenv\Scripts\activate

### ✅ 3. Install dependencies:

pip install -r requirements.txt


### ✅ 4. Run the Streamlit app:

streamlit run app.py


### ✅ 5. Open your browser at:

http://localhost:8501



🔍 Sample Queries
What are the rules for selling motor vehicles on eBay?

# Can I cancel a transaction as a buyer?

# What happens if I violate eBay's terms?

# Is arbitration mandatory for disputes?


🧠 How it Works
✅ The document is chunked into 100–300 word segments using sentence-aware splitting.

✅ Each chunk is embedded into a vector using all-MiniLM-L6-v2.

✅ FAISS vector DB is used to retrieve top matching chunks based on the user query.

✅ Retrieved chunks + user question are injected into a prompt.

✅ The LLM (google/flan-t5-large) generates an answer grounded in the context.


⚠️ Limitations
# Model responses depend on the relevance of retrieved chunks.

# Answers are limited to the knowledge present in the provided document.

# Some long or vague queries may produce less accurate results.


📜 License
This project is created as part of the Amlgo Labs Junior AI Engineer assignment.


🙋‍♂️ Author
Naushad Alam
Email: naushadlil01@gmail.com
GitHub: github.com/Naushad-Alam01