import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit.components.v1 as components
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import ast

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/atomic_habits_analysis.csv")
    df["Top Keywords"] = df["Top Keywords"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

# Set layout
st.set_page_config(page_title="Atomic Habits NLP Dashboard", layout="wide")
st.sidebar.title("📘 Atomic Habits NLP Dashboard")
section = st.sidebar.radio("Navigate", ["📚 Chapter Insights", "🕸️ Concept Network", "ℹ️ About"])

df = load_data()

# ========================================
# 📚 CHAPTER INSIGHTS
# ========================================
if section == "📚 Chapter Insights":
    st.title("📚 Chapter-wise Insights")

    # Chapter selector
    st.sidebar.header("📖 Filter Chapter")
    chapter_selected = st.sidebar.selectbox("Choose a Chapter", df["Chapter"])

    row = df[df["Chapter"] == chapter_selected].iloc[0]

    # Summary
    st.subheader(f"🧠 Summary - Chapter {chapter_selected}")
    with st.expander("🔎 View Full Summary"):
        st.write(row["Summary"])

    # Sentiment display (colored)
    sentiment = row["Sentiment"]
    sent_color = "green" if sentiment > 0.1 else "orange" if sentiment > 0 else "red"
    st.markdown(f"**Sentiment Score:** <span style='color:{sent_color}'><b>{round(sentiment, 3)}</b></span>", unsafe_allow_html=True)

    # Top Keywords
    st.markdown("**🔑 Top Keywords:**")
    st.write("• " + "\n• ".join(row["Top Keywords"]))

    # WordCloud for entire book
    st.subheader("🌥 WordCloud for Entire Book")
    all_keywords = sum(df["Top Keywords"].tolist(), [])
    wc = WordCloud(width=1000, height=400, background_color='white').generate(" ".join(all_keywords))
    plt.figure(figsize=(12, 4))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    # --- Overall Sentiment Chart with Emoji ---
    st.subheader("📊 Overall Sentiment per Chapter")

    def sentiment_emoji(score):
        if score > 0.1:
            return "😊"
        elif score > 0:
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
    ax.set_ylabel("Chapter", fontsize=12)
    ax.set_title("📘 Sentiment Score by Chapter (Red ➝ Green)", fontsize=14)

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical")
    cbar.set_label("Sentiment Polarity", fontsize=10)
    st.pyplot(fig)

    # CSV Export
    st.download_button("⬇️ Download Insights CSV", df.to_csv(index=False), file_name="atomic_habits_analysis.csv")

# ========================================
# 🕸️ CONCEPT NETWORK
# ========================================
elif section == "🕸️ Concept Network":
    st.title("🕸️ Concept Relationship Network")
    st.write("Explore how core ideas like habit, cue, reward, etc. relate visually.")

    try:
        with open("graph/atomic_habits_concept_network.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=600, scrolling=True)
    except FileNotFoundError:
        st.error("❌ Concept network file not found. Place it at `graph/atomic_habits_concept_network.html`.")

# ========================================
# ℹ️ ABOUT
# ========================================
elif section == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
This interactive dashboard analyzes the book *Atomic Habits* by **James Clear** using NLP:

- 📘 Extracted summaries of each chapter  
- 📈 Performed sentiment analysis  
- 🔑 Identified top keywords using TF-IDF  
- 🕸️ Built a concept relationship network  
- 📊 Created an overall sentiment map + wordcloud

**Built with:** Python · Streamlit · Transformers · PyVis  
""")
