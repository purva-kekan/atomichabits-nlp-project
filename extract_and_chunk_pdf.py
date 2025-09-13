import fitz  # PyMuPDF
import re
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
nltk.download("punkt")

# === Step 1: Extract text from PDF ===
pdf_path = "C:/Users/purva/OneDrive/Desktop/PERSONAL PROJECT/NLP_Analysis_Atomic_Habits/semantic_search_atomic_habits/data/Atomic habits.pdf" 

# Combine text from all pages
doc = fitz.open(pdf_path)  
raw_text = " ".join(page.get_text() for page in doc)

# Clean up whitespace
clean_text = re.sub(r'\s+', ' ', raw_text)

# === Step 2: Chunk the text ===
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

chunks = chunk_text(clean_text)

# === Step 3: Save to CSV ===
df = pd.DataFrame({"chunk_id": range(len(chunks)), "text": chunks})
df.to_csv("C:/Users/purva/OneDrive/Desktop/PERSONAL PROJECT/NLP_Analysis_Atomic_Habits/semantic_search_atomic_habits/data/full_text_chunks.csv", index=False)

print(f"✅ Extracted and saved {len(chunks)} chunks to data/full_text_chunks.csv")
