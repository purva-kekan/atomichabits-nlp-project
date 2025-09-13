import os
import json
import pandas as pd
import fitz  # PyMuPDF
import re
from nltk.tokenize import sent_tokenize

# Step 1: Load book metadata
with open("books.json", "r") as f:
    books = json.load(f)

# process first book marked as 'ready'
book = next((b for b in books if b["status"] == "ready"), None)

if not book:
    print("❌ No book marked as 'ready' in books.json.")
    exit()

book_id = book["id"]
book_path = os.path.join("data", book["filename"])
output_path = os.path.join("processed", f"{book_id}.csv")

print(f"📘 Processing book: {book['title']}")

# Step 2: Extract text from PDF
doc = fitz.open(book_path)
raw_text = " ".join(page.get_text() for page in doc)

# clean up extra spaces
raw_text = re.sub(r'\s+', ' ', raw_text)

# Step 3: Chunk into sentences
def chunk_text(text, chunk_size=1000):
    sentences = sent_tokenize(text)
    chunks, chunk = [], ""
    for sentence in sentences:
        if len(chunk) + len(sentence) < chunk_size:
            chunk += sentence + " "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

chunks = chunk_text(raw_text)

# Step 4: Save to CSV
df = pd.DataFrame({"chunk_id": range(len(chunks)), "text": chunks})
df.to_csv(output_path, index=False)

print(f"✅ Saved {len(chunks)} chunks to {output_path}")
