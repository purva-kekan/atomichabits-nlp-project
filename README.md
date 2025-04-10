# 📘 Atomic Habits NLP Dashboard

An interactive dashboard that analyzes the book **_Atomic Habits_** by *James Clear* using Natural Language Processing. This project provides chapter-wise insights, sentiment scores, top keywords, and a concept relationship network that helps visualize how ideas like _habit_, _cue_, _reward_, and _identity_ are connected throughout the book.

---

## 🚀 Features

### 📚 Chapter Insights
- Extracted summaries using Transformer models (BART/DistilBART)
- Performed sentiment analysis using TextBlob
- Extracted top keywords with TF-IDF
- Visualized keywords using a WordCloud
- Emoji-enhanced sentiment polarity chart per chapter
- Downloadable CSV with all insights

### 🕸️ Concept Network
- Visualizes the relationship between key behavioral science concepts
- Built using PyVis and NetworkX
- Fully interactive and embeddable in the Streamlit app

---

## 📈 Visual Previews

### 🌥 WordCloud of All Chapters

Visualizes the most important recurring terms across the entire book.

### 📊 Sentiment Analysis Chart

Emoji-coded chart showing whether each chapter is generally positive (😊), neutral (😐), or negative (😞).

### 🕸️ Interactive Concept Map

Built with PyVis to show how behavioral science concepts are interconnected across chapters. You can explore:

- Habit loops (cue → craving → response → reward)
- Identity-based habit building
- Role of environment and routine in behavior change

---

## 🛠️ Tech Stack

- **NLP & Processing**: TextBlob, Transformers, TF-IDF
- **Visualization**: Streamlit, PyVis, WordCloud, Matplotlib
- **Language**: Python 3

---

## 📚 Acknowledgements

- Book: _Atomic Habits_ by James Clear
- Streamlit – for building the dashboard
- PyVis & NetworkX – for network graph visualizations

---

## 📄 License

This project is for educational and non-commercial use only.  
All rights to the original book content belong to the author and publisher.
