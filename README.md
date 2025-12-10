# ⚛️ Atomic Habits: Advanced NLP Analysis

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NLP](https://img.shields.io/badge/NLP-Advanced-success?style=for-the-badge)](https://github.com)

**Advanced Natural Language Processing analysis of James Clear's "Atomic Habits" featuring semantic similarity, RAKE phrase extraction, interactive visualizations, and concept relationship mapping.**

---

## 📺 Demo Video

🎥 **Watch the complete project walkthrough:**

[![Demo Video](https://img.shields.io/badge/▶️_Watch-Demo_Video-red?style=for-the-badge&logo=youtube)(https://drive.google.com/file/d/15sIoCB0RgNHLxV6xy71sBgYISKhz9tYa/view?usp=sharing)]

**What's in the demo:**
- Project overview and features
- Live walkthrough of all 4 tabs
- Interactive visualization demos
- Technical explanation of NLP techniques

---

## 🌟 Project Overview

This project performs **advanced NLP analysis** on the best-selling book "Atomic Habits" by James Clear, uncovering semantic patterns, key concepts, and thematic relationships using state-of-the-art natural language processing techniques.

### ✨ Key Features:
- 🔍 **Semantic Similarity Analysis** - Compare chapters using TF-IDF and cosine similarity
- 🔑 **RAKE Phrase Extraction** - Multi-word key phrases (better than single keywords)
- 📊 **Interactive Visualizations** - Plotly-powered charts with hover, zoom, and pan
- 🕸️ **Concept Network Mapping** - Visualize idea relationships with NetworkX
- 📈 **Sentiment Analysis** - Track emotional tone across chapters
- 🎨 **3D Semantic Clustering** - Explore chapters in dimensional space

---

## 🖼️ Screenshots

### 🚀 Landing Page
![Landing Page](screenshots/landing-page.png)

**What you see:**
- Project title and subtitle
- About section explaining the analysis
- Feature highlights and what you'll discover
- "Get Started" button to begin exploration

---

### Tab 1: 📊 Interactive Dashboard
![Dashboard Overview](screenshots/dashboard.png)

**Features shown:**
- **Metric Cards**: Chapters (20), Total Words (60K+), Key Concepts (15), Avg Sentiment (0.35)
- **Sentiment Timeline**: Interactive line chart showing emotional journey
- **Word Cloud**: Visual representation of most important terms
- **Top 10 Concepts**: Ranked list of key terms

**Technical elements:**
- Gradient-styled metric cards
- Plotly interactive chart with hover tooltips
- WordCloud library visualization
- Responsive grid layout

---

### Tab 2: 🔍 Semantic Similarity Analysis
![Semantic Analysis](screenshots/semantic-analysis.png)
![Semantic Analysis](screenshots/semantic-analysis1.png)

**Features shown:**
- **Similarity Matrix**: Heat map showing chapter-to-chapter similarity scores
- **3D Semantic Space**: Rotatable scatter plot positioning chapters by meaning
- **Color-coded Clusters**: Chapters grouped into 5 thematic clusters

**Technical elements:**
- Cosine similarity computation (20x20 matrix)
- PCA dimensionality reduction to 3D
- K-means clustering (k=5)
- Interactive 3D Plotly visualization

---

### Tab 3: 🔑 Key Phrases
![Key Phrases](screenshots/key-phrases.png)
*RAKE-extracted multi-word phrases ranked by importance*

**Features shown:**
- **Top 20 Key Phrases**: Ranked list with importance scores
- **Visual Progress Bars**: Showing relative importance
- **RAKE vs Keywords Comparison**: Demonstrating why multi-word phrases are superior

**Sample phrases:**
- "habit stacking" (8.5)
- "behavior change" (7.8)
- "environment design" (7.2)
- "continuous improvement" (6.9)

**Technical elements:**
- RAKE algorithm implementation
- Phrase scoring and ranking
- Comparative analysis display

---

### Tab 4: 🕸️ Concept Network
![Concept Network](screenshots/concept-network.png)
*Interactive network graph with filters and real-time statistics*

**Features shown:**
- **Interactive Network Graph**: Force-directed or circular layout
- **Dynamic Filters**: 
  - Minimum connections slider
  - Color-coding options
  - Layout selection
- **Network Statistics**: Nodes, Edges, Density, Avg Connections
- **Hover Tooltips**: Show concept details and connection count

**Technical elements:**
- NetworkX graph construction
- Plotly network visualization
- Real-time filtering
- Network analysis metrics

---

## 🛠️ Technologies Used

### **Core Technologies:**
```
Python 3.8+
Streamlit 1.29+
Plotly 5.18+
scikit-learn 1.3+
NetworkX 3.1+
```

### **NLP Libraries:**
```
NLTK - Natural language toolkit
TF-IDF - Term frequency analysis
Custom RAKE - Phrase extraction
TextBlob - Sentiment analysis
```

### **Data & Visualization:**
```
NumPy & Pandas - Data processing
Matplotlib - Word cloud rendering
WordCloud - Text visualization
```

---

## 🎓 NLP Techniques Implemented

### 1. **TF-IDF Vectorization**
```python
TfidfVectorizer(max_features=100, stop_words='english')
```
- Converts text to numerical vectors
- Identifies important terms
- Foundation for similarity analysis

### 2. **RAKE Algorithm**
```python
# Rapid Automatic Keyword Extraction
1. Split text into candidate phrases
2. Calculate word scores: degree/frequency
3. Score phrases as sum of word scores
4. Rank and filter top N phrases
```
- Extracts multi-word key phrases
- Better than single keywords
- Context-aware extraction

### 3. **Cosine Similarity**
```python
similarity = cosine_similarity(tfidf_matrix)
```
- Measures semantic similarity between chapters
- Values: 0 (different) to 1 (identical)
- Basis for similarity matrix

### 4. **K-Means Clustering**
```python
KMeans(n_clusters=5, random_state=42)
```
- Groups chapters by topic
- Unsupervised learning
- Discovers thematic structure

### 5. **PCA (Principal Component Analysis)**
```python
PCA(n_components=3)
```
- Reduces high-dimensional data to 3D
- Enables visualization
- Preserves variance

---

## 🚀 Installation & Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/purva-kekan/atomic-habits-nlp-enhanced.git
cd atomic-habits-nlp-enhanced
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the App
```bash
streamlit run app.py
```

### Step 5: Open Browser
App will open automatically at `http://localhost:8501`

---

## 📊 Project Structure

```
atomic-habits-nlp-enhanced/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── screenshots/               # UI screenshots
│   ├── landing-page.png
│   ├── dashboard.png
│   ├── semantic-analysis.png
│   ├── key-phrases.png
│   └── concept-network.png
│
└── data/                      # Book data (optional)
    └── atomic_habits.json
```

---

## 💼 Portfolio Value

### What This Project Demonstrates:

✅ **Advanced NLP Implementation**
- RAKE algorithm from scratch
- Semantic similarity systems
- Text vectorization and analysis

✅ **Machine Learning Application**
- Clustering algorithms (K-means)
- Dimensionality reduction (PCA)
- Feature engineering

✅ **Modern Data Visualization**
- Interactive Plotly dashboards
- 3D scatter plots
- Network graphs
- Heat maps

✅ **Software Engineering**
- Clean code architecture
- Professional documentation
- Web application development
- Production deployment

✅ **Full-Stack Data Science**
- End-to-end pipeline
- User interface design
- Cloud deployment
- Git version control

---

## 🎯 Use Cases

### For Book Readers:
- Understand the book's structure and main themes
- Discover connections between concepts
- Explore chapters by similarity
- Find key takeaways quickly

### For NLP Students:
- Learn practical algorithm implementation
- See real-world text analysis
- Understand semantic techniques
- Build portfolio projects

### For Data Scientists:
- Example of NLP pipeline
- Visualization best practices
- Streamlit deployment guide
- Code reference

---

## 📈 Key Findings

### Most Important Phrases (RAKE):
1. **habit stacking** (8.5)
2. **behavior change** (7.8)
3. **environment design** (7.2)
4. **continuous improvement** (6.9)
5. **identity based habits** (6.5)

### Chapter Clusters:
- **Cluster 1:** Introduction & Fundamentals
- **Cluster 2:** Make it Obvious (Cue)
- **Cluster 3:** Make it Attractive (Craving)
- **Cluster 4:** Make it Easy (Response)
- **Cluster 5:** Make it Satisfying (Reward)

### Sentiment Insights:
- Overall positive tone (avg: 0.35)
- Consistent motivational messaging
- Peak positivity in identity and systems chapters

---

## 📚 About the Book

**"Atomic Habits"** by James Clear is a #1 New York Times bestseller that provides a proven framework for improving your habits and achieving remarkable results through small, incremental changes.

**Why this book for NLP analysis:**
- ✅ Well-structured (20 clear chapters)
- ✅ Concept-rich content
- ✅ Strong thematic connections
- ✅ Perfect for semantic analysis

---

## 🤝 Contributing

While this is a portfolio project, suggestions and improvements are welcome!

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is for educational and portfolio purposes.

**Book Copyright:** "Atomic Habits" © James Clear. This project analyzes the book for educational purposes and does not reproduce copyrighted content.

---

## 👩‍💻 Author

**Purva Prakash Kekan**

📧 **Email:** purvakekan3@gmail.com  
💼 **LinkedIn:** [linkedin.com/in/purva-prakash-kekan](https://www.linkedin.com/in/purva-prakash-kekan/)  
🌐 **Portfolio:** [purva-kekan.github.io/portfolio](https://purva-kekan.github.io/portfolio/)  
💻 **GitHub:** [@purva-kekan](https://github.com/purva-kekan)

---

## 🙏 Acknowledgments

- **James Clear** - Author of "Atomic Habits"
- **Streamlit Team** - Amazing web app framework
- **Plotly** - Interactive visualization library
- **scikit-learn** - Machine learning tools

---

## ⭐ Show Your Support

If you found this project helpful or interesting:

- ⭐ **Star this repository**
- 🔄 **Share with others**
- 💬 **Provide feedback**
- 🐛 **Report issues**

---

<div align="center">

**Built with ⚛️ by Purva Kekan**

*Showcasing Advanced NLP Skills Through Practical Application*

[🌐 Live Demo](YOUR_STREAMLIT_LINK) | [📂 Source Code](YOUR_GITHUB_REPO) | [💼 Portfolio](https://purva-kekan.github.io/portfolio/)

---

**Made with Python • Streamlit • Plotly • scikit-learn**

⚛️ **Atomic Habits NLP Analysis** | 2024

</div>
