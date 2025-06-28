import streamlit as st
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --------------------
# ✅ Load Models
# --------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return embedder, generator

embedder, generator = load_models()

# --------------------
# ✅ Load FAISS & Chunks
# --------------------
index = faiss.read_index("vectordb/index.faiss")
chunks = np.load("vectordb/chunks.npy", allow_pickle=True)


# --------------------
# ✅ Retriever Function
# --------------------
def retrieve_relevant_chunks(query, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, k)
    retrieved = [chunks[i] for i in I[0]]
    return retrieved


# --------------------
# ✅ Prompt Builder
# --------------------
def build_prompt(retrieved_chunks, user_query):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question: {user_query}

Answer:"""
    return prompt


# --------------------
# ✅ Generator Function
# --------------------
def generate_answer(prompt):
    response = generator(
        prompt,
        max_length=512,
        do_sample=False,
        temperature=0.2,
    )
    return response[0]['generated_text']


# --------------------
# ✅ Streamlit App
# --------------------
st.set_page_config(page_title="RAG Chatbot - eBay User Agreement", layout="wide")
st.title("🧠 RAG Chatbot — eBay User Agreement")

st.sidebar.title("📄 Chatbot Info")
st.sidebar.markdown("**Model:** google/flan-t5-large")
st.sidebar.markdown(f"**Total Chunks:** {len(chunks)}")
st.sidebar.markdown("**Vector DB:** FAISS")
st.sidebar.button("🔄 Clear Chat", on_click=lambda: st.session_state.clear())

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat history
for msg in st.session_state['messages']:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Type your question about eBay's User Agreement..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state['messages'].append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching..."):
            # RAG Pipeline
            retrieved_chunks = retrieve_relevant_chunks(prompt, k=3)
            built_prompt = build_prompt(retrieved_chunks, prompt)
            response = generate_answer(built_prompt)

            # Streaming simulation (sentence by sentence)
            output_placeholder = st.empty()
            output_text = ""
            for sentence in response.split('. '):
                output_text += sentence.strip() + ". "
                output_placeholder.markdown(output_text.strip())
                time.sleep(0.2)  # small delay for effect

            output_placeholder.markdown(output_text.strip())
            st.session_state['messages'].append({"role": "assistant", "content": output_text.strip()})

        with st.expander("🔗 Source Chunks"):
            for idx, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {idx + 1}:** {chunk}")
