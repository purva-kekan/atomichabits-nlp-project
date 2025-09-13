import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit.components.v1 as components
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import ast
import json
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# ========== Load book metadata ==========
with open("books.json", "r") as f:
    books = json.load(f)
book = next((b for b in books if b["status"] == "ready"), None)
book_id = book["id"]
book_title = book["title"]

# ========== App Layout ==========
st.set_page_config(page_title="Self-Help NLP Platform", layout="wide")
st.sidebar.title(f"📘 NLP Explorer – {book_title}")
section = st.sidebar.radio("Navigate", ["📚 Chapter Insights", "🔍 Semantic Search", "🕸️ Concept Network", "ℹ️ About"])

# ========== 📚 CHAPTER INSIGHTS ==========
if section == "📚 Chapter Insights":
    st.title("📚 Chapter-wise Insights")

    df = pd.read_csv(f"data/atomic_habits_analysis.csv")
    df["Top Keywords"] = df["Top Keywords"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


    st.sidebar.header("📖 Filter Chapter")
    chapter_selected = st.sidebar.selectbox("Choose a Chapter", df["Chapter"])

    row = df[df["Chapter"] == chapter_selected].iloc[0]
    st.subheader(f"🧠 Summary - Chapter {chapter_selected}")
    with st.expander("🔎 View Full Summary"):
        st.write(row["Summary"])

    # Show Sentiment Label for this Chapter
    score = row["Sentiment"]  # assuming your column is named "Sentiment"

    def get_sentiment_label(score):
        if score > 0.1:
            return "😊" "Positive"
        elif score > 0.05:
            return "😐" "Neutral"
        else:
            return "😞" "Negative"

    st.markdown(f"**🧭 Chapter Sentiment:** {get_sentiment_label(score)}")
    st.markdown("**🔑 Top Keywords:**")
    st.write("• " + "\n• ".join(row["Top Keywords"]))

    st.subheader("🌥 WordCloud")
    all_keywords = sum(df["Top Keywords"].tolist(), [])
    wc = WordCloud(width=1000, height=400, background_color='white').generate(" ".join(all_keywords))
    plt.figure(figsize=(12, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    st.subheader("📊 Overall Sentiment per Chapter")

    def sentiment_emoji(score):
        if score > 0.1:
            return "😊"
        elif score > 0.05:
            return "😐"
        else:
            return "😞"

    df["Label"] = df["Chapter"].astype(str) + " " + df["Sentiment"].apply(sentiment_emoji)
    norm = plt.Normalize(df["Sentiment"].min(), df["Sentiment"].max())
    colors = plt.cm.RdYlGn(norm(df["Sentiment"]))

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(df["Label"], df["Sentiment"], color=colors)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Sentiment Polarity", fontsize=12)
    ax.set_title("📘 Sentiment Score by Chapter (Red ➝ Green)", fontsize=14)
    st.pyplot(fig)

    #st.download_button("⬇️ Download Insights CSV", df.to_csv(index=False), file_name="chapter_sentiment_summary.csv")

# ========== 🔍 SEMANTIC SEARCH ==========
elif section == "🔍 Semantic Search":
    st.title("🔍 Ask the Book")

    query = st.text_input("Ask something from the book:")
    top_k = st.slider("Number of results to show", 1, 5, 3)
    st.markdown("Higher similarity means better match to your question.")


    if query:
        index_path = f"embeddings/{book_id}_index.bin"
        meta_path = f"embeddings/{book_id}_metadata.pkl"

        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            chunks = pickle.load(f)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vec = model.encode([query])
        scores, indices = index.search(np.array(query_vec), k=top_k)

        results = []

        for i, distance in zip(indices[0], scores[0]):
            similarity = 1/ (1 + distance)
            results.append((similarity, chunks[i]))

        results.sort(reverse=True, key=lambda x: x[0])

        #display results
        for sim, chunk in results:
            st.write(f"**Similarity:** {sim:.2%}")
            st.write(chunk)
            st.markdown("---")
        

# ========== 🕸️ CONCEPT NETWORK ==========
elif section == "🕸️ Concept Network":
    st.title("🕸️ Concept Relationship Graph")
    try:
        with open("graph/atomic_habits_concept_network.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600, scrolling=True)
    except FileNotFoundError:
        st.error("❌ Graph not found. Place the HTML file in `graph/`.")

# ========== ℹ️ ABOUT ==========
elif section == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
This app is a multi-book NLP platform for exploring self-help books using AI.

**Features:**
- 📚 Chapter-wise sentiment + keywords
- 🔍 Semantic Search using sentence embeddings + FAISS
- 🕸️ Concept graphs to visualize key ideas
- 🧠 Chatbot & quote audio coming soon!

**Built with:** Streamlit · SentenceTransformers · FAISS · PyMuPDF · PyVis  
    """)
