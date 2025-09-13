import streamlit as st
import json
import pandas as pd
import numpy as np
import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer

# === Hugging Face API Key from .streamlit/secrets.toml ===
HF_API_KEY = st.secrets["HF_API_KEY"]
headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# === Hugging Face Q&A Model ===
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# === Load Book Info ===
with open("books.json", "r") as f:
    books = json.load(f)

book = next((b for b in books if b["status"] == "ready"), None)
book_id = book["id"]
book_title = book["title"]

st.set_page_config(page_title="📘 Smart Book Q&A", layout="wide")
st.title(f"💬 Ask Anything About '{book_title}'")

# === Load FAISS Index + Chunks ===
try:
    index = faiss.read_index(f"embeddings/{book_id}_index.bin")
    with open(f"embeddings/{book_id}_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
except Exception as e:
    st.error(f"❌ Could not load index or embeddings: {e}")
    st.stop()

# === Embed Model for Semantic Search ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Hugging Face LLM Call ===
def ask_with_context(question, context):
    prompt = f"""
<|system|>
You are a helpful assistant answering questions using only the given context.
<|user|>
Question: {question}

Context:
{context}
<|assistant|>
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.5,
            "max_new_tokens": 300
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0].get("generated_text", "⚠️ No answer returned.")
    else:
        return f"❌ API Error {response.status_code}: {response.text}"

# === Streamlit UI ===
query = st.text_input("Ask your question:")
if st.button("Get Answer") and query:
    with st.spinner("Retrieving answer..."):

        # 1. Embed the query
        q_vec = embedder.encode([query])
        scores, indices = index.search(np.array(q_vec), k=6)

        # 2. Get top chunks
        context_chunks = [chunks[i] for i in indices[0]]
        context = "\n\n".join(context_chunks)

        # 3. Send to Hugging Face
        answer = ask_with_context(query, context)

        # 4. Show result
        st.caption("📘 Based on top 6 relevant parts of the book.")
        st.markdown("### 📘 Answer:")
        st.write(answer)
