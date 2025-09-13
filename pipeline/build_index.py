import os
import json
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# ========== Step 1: Load book metadata ==========
with open("books.json", "r") as f:
    books = json.load(f)

book = next((b for b in books if b["status"] == "ready"), None)

if not book:
    print("❌ No book marked as 'ready' in books.json.")
    exit()

book_id = book["id"]
csv_path = os.path.join("processed", f"{book_id}.csv")
output_index_path = os.path.join("embeddings", f"{book_id}_index.bin")
output_meta_path = os.path.join("embeddings", f"{book_id}_metadata.pkl")

print(f"📚 Building embeddings for: {book['title']}")

# ========== Step 2: Load text chunks ==========
df = pd.read_csv(csv_path)
chunks = df["text"].tolist()

# ========== Step 3: Generate embeddings ==========
print("🔄 Generating sentence embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks, show_progress_bar=True)

# ========== Step 4: Build FAISS index ==========
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ========== Step 5: Save the index and metadata ==========
faiss.write_index(index, output_index_path)

with open(output_meta_path, "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Saved FAISS index to {output_index_path}")
print(f"✅ Saved metadata to {output_meta_path}")
