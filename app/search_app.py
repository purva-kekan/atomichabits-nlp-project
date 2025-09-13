import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import os
import json

# ========== Step 1: Load book info ==========
with open("books.json", "r") as f:
    books = json.load(f)

book = next((b for b in books if b["status"] == "ready"), None)
book_id = book["id"]
st.title(f"🔍 Semantic Search — {book['title']}")

# ========== Step 2: Load index & metadata ==========
index_path = f"embeddings/{book_id}_index.bin"
meta_path = f"embeddings/{book_id}_metadata.pkl"

index = faiss.read_index(index_path)

with open(meta_path, "rb") as f:
    chunks = pickle.load(f)

# Load the same model used for embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

# ========== Step 3: Search interface ==========
query = st.text_input("Ask something from the book:")

top_k = st.slider("How many results to show?", 1, 10, 3)

if query:
    query_vec = model.encode([query])
    scores, indices = index.search(np.array(query_vec), k=top_k)

    st.markdown("### 🔎 Top Matching Passages")
    for i, score in zip(indices[0], scores[0]):
        st.write(f"**Score:** {score:.2f}")
        st.write(chunks[i])
        st.markdown("---")
