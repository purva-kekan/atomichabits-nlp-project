import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from pyvis.network import Network

# Load chunked book
df = pd.read_csv("data/full_text_chunks.csv")

# --- Step 1: Extract Concepts ---
vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
tfidf_matrix = vectorizer.fit_transform(df["text"])
terms = vectorizer.get_feature_names_out()

# --- Step 2: Create Concept Similarity Matrix ---
similarity = cosine_similarity(tfidf_matrix.T)

# --- Step 3: Build Graph ---
G = nx.Graph()
for i in range(len(terms)):
    G.add_node(terms[i])
    for j in range(i+1, len(terms)):
        if similarity[i][j] > 0.15:  # adjust threshold
            G.add_edge(terms[i], terms[j], weight=similarity[i][j])

# --- Step 4: Visualize with PyVis ---
net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
net.from_nx(G)
net.force_atlas_2based()
net.show("graph/atomic_habits_concept_network.html")

print("✅ Concept graph saved to graph/atomic_habits_concept_network.html")
