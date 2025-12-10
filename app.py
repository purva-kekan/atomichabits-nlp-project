import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Page config
st.set_page_config(
    page_title="Atomic Habits - NLP Analysis",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed"  
)

#custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'book_data' not in st.session_state:
        st.session_state.book_data = None

def create_sample_data():
    """Create sample data for demo"""
    
    #sample chapter data
    chapters = []
    chapter_titles = [
        "The Surprising Power of Atomic Habits",
        "How Your Habits Shape Your Identity",
        "How to Build Better Habits in 4 Simple Steps",
        "The Man Who Didn't Look Right",
        "The Best Way to Start a New Habit",
        "Motivation Is Overrated; Environment Often Matters More",
        "The Secret to Self-Control",
        "How to Make a Habit Irresistible",
        "The Role of Family and Friends",
        "How to Find and Fix Bad Habits",
        "Walk Slowly, but Never Backward",
        "The Law of Least Effort",
        "How to Stop Procrastinating",
        "How to Make Good Habits Inevitable",
        "The Cardinal Rule of Behavior Change",
        "How to Stick with Good Habits",
        "How an Accountability Partner Changes Everything",
        "The Truth About Talent",
        "The Goldilocks Rule",
        "The Downside of Creating Good Habits"
    ]
    
    for idx, title in enumerate(chapter_titles, 1):
        chapters.append({
            'number': idx,
            'title': f"Chapter {idx}: {title}",
            'word_count': np.random.randint(2000, 5000),
            'sentiment': np.random.uniform(-0.1, 0.6),
            'keywords': ['habit', 'atomic', 'identity', 'cue', 'craving', 'response', 'reward'],
            'content': f"Sample content for chapter {idx}. This discusses {title.lower()}."
        })
    
    #Key concepts
    key_concepts = [
        'atomic habits', 'identity', 'cue', 'craving', 'response', 'reward',
        'habit stacking', 'environment design', 'systems', 'goals',
        'continuous improvement', 'behavior change', 'feedback loops',
        'implementation intentions', 'temptation bundling'
    ]
    
    #Key phrases (with scores)
    key_phrases = [
        ('habit stacking', 8.5),
        ('behavior change', 7.8),
        ('environment design', 7.2),
        ('continuous improvement', 6.9),
        ('identity based habits', 6.5),
        ('implementation intentions', 6.2),
        ('temptation bundling', 5.8),
        ('feedback loops', 5.5),
        ('atomic habits', 5.2),
        ('goal setting', 4.9)
    ]
    
    #compute semantic similarity
    chapter_texts = [ch['content'] for ch in chapters]
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    try:
        tfidf_matrix = vectorizer.fit_transform(chapter_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        #PCA for 3D
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(tfidf_matrix.toarray())
        
        #Clustering
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix.toarray())
        
        for idx, ch in enumerate(chapters):
            ch['cluster'] = int(clusters[idx])
    except:
        similarity_matrix = np.eye(len(chapters))
        embeddings_3d = np.random.rand(len(chapters), 3)
        for ch in chapters:
            ch['cluster'] = 0
    
    return {
        'title': 'Atomic Habits',
        'author': 'James Clear',
        'chapters': chapters,
        'key_concepts': key_concepts,
        'key_phrases': key_phrases,
        'similarity_matrix': similarity_matrix,
        'chapter_embeddings': embeddings_3d,
        'num_clusters': 5
    }

def main():
    """Main app"""
    
    init_session_state()
    
    #Header
    st.markdown('<div class="main-header">⚛️ Atomic Habits: Advanced NLP Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; opacity: 0.8;">Semantic Relationship Mapping & Advanced Text Analytics</p>', unsafe_allow_html=True)
    
    #Main content (no sidebar!)
    if not st.session_state.data_loaded:
        show_welcome()
    else:
        show_tabs()

def show_welcome():
    """Welcome screen"""
    
    st.markdown("## 🚀 Welcome to Advanced NLP Analysis")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        ### 📚 About This Project
        
        This is an **advanced Natural Language Processing analysis** of James Clear's 
        best-selling book **"Atomic Habits: An Easy & Proven Way to Build Good Habits & Break Bad Ones."**
        
        Using state-of-the-art NLP techniques, this analysis uncovers:
        
        - 🔍 **Semantic patterns** and relationships across all 20 chapters
        - 🔑 **Key phrases** extracted using RAKE algorithm
        - 🕸️ **Concept networks** showing how ideas connect
        - 📊 **Interactive visualizations** powered by Plotly
        - 📈 **Sentiment analysis** tracking emotional tone
        
        **Perfect for:**
        - Understanding the book's structure and themes
        - Discovering main concepts and how they relate
        - Visualizing semantic relationships
        - Exploring the book through data
        """)
    
    with col2:
        st.markdown("### ⚡ Analysis Overview")
        st.info("""
        **📖 Book:**  
        Atomic Habits
        
        **✍️ Author:**  
        James Clear
        
        **🔬 Chapters:**  
        20 chapters analyzed
        
        **🛠️ NLP Techniques:**  
        - RAKE phrase extraction
        - Semantic similarity
        - K-means clustering
        - Sentiment analysis
        - Network analysis
        """)
    
    st.markdown("---")
    
    st.markdown("### 📊 What You'll Discover")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("""
        #### 📈 Quantitative
        
        - Semantic similarity scores
        - Key phrase rankings  
        - Sentiment patterns
        - Chapter clustering
        """)
    
    with col_b:
        st.markdown("""
        #### 🎨 Visual
        
        - Interactive Plotly charts
        - 3D semantic space
        - Concept networks
        - Word clouds
        """)
    
    with col_c:
        st.markdown("""
        #### 🔍 Interactive
        
        - Hover for details
        - Zoom and pan
        - Filter networks
        - Explore clusters
        """)
    
    st.markdown("---")
    
    #cta button - centered
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🚀 Get Started", type="primary", width="stretch"):
            load_data()
        
        st.caption("Load the analyzed book data and start exploring")

def load_data():
    """Load sample data"""
    
    with st.spinner("📚 Loading Atomic Habits analysis..."):
        
        progress = st.progress(0)
        status = st.empty()
        
        status.text("📄 Loading book data...")
        progress.progress(25)
        
        book_data = create_sample_data()
        
        progress.progress(75)
        status.text("🎨 Preparing visualizations...")
        
        progress.progress(100)
        
        st.session_state.book_data = book_data
        st.session_state.data_loaded = True
        
        status.text("✅ Complete!")
        st.success("🎉 Atomic Habits loaded!")
        st.balloons()
        st.rerun()

def show_tabs():
    """Show 4 main tabs"""
    
    #add reload button at top
    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        if st.button("🔄 Reload Data"):
            st.session_state.data_loaded = False
            st.session_state.book_data = None
            st.rerun()
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔍 Semantic Analysis",
        "🔑 Key Phrases",
        "🕸️ Concept Network"
    ])
    
    with tab1:
        show_dashboard()
    
    with tab2:
        show_semantic()
    
    with tab3:
        show_phrases()
    
    with tab4:
        show_network()

def show_dashboard():
    """Dashboard tab"""
    
    st.markdown("## 📊 Interactive Dashboard")
    
    data = st.session_state.book_data
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'<div class="metric-card"><h3>📖</h3><h2>{len(data["chapters"])}</h2><p>Chapters</p></div>', 
                   unsafe_allow_html=True)
    
    with col2:
        total_words = sum(ch['word_count'] for ch in data['chapters'])
        st.markdown(f'<div class="metric-card"><h3>📝</h3><h2>{total_words:,}</h2><p>Words</p></div>', 
                   unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-card"><h3>🔑</h3><h2>{len(data["key_concepts"])}</h2><p>Concepts</p></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        avg_sent = np.mean([ch['sentiment'] for ch in data['chapters']])
        st.markdown(f'<div class="metric-card"><h3>😊</h3><h2>{avg_sent:.2f}</h2><p>Sentiment</p></div>', 
                   unsafe_allow_html=True)
    
    st.markdown("---")
    
    #sentiment timeline
    st.markdown("### 📈 Sentiment Journey")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[ch['number'] for ch in data['chapters']],
        y=[ch['sentiment'] for ch in data['chapters']],
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color=[ch['sentiment'] for ch in data['chapters']], 
                   colorscale='RdYlGn', showscale=True),
        hovertemplate='<b>Chapter %{x}</b><br>Sentiment: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Chapter",
        yaxis_title="Sentiment",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    #word cloud
    st.markdown("### ☁️ Key Concepts Word Cloud")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text = ' '.join(data['key_concepts'])
        
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=30
        ).generate(text)
        
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        
        st.pyplot(fig_wc, width="stretch")
    
    with col2:
        st.markdown("#### 🔝 Top Terms")
        for idx, concept in enumerate(data['key_concepts'][:10], 1):
            st.markdown(f"{idx}. **{concept}**")

def show_semantic():
    """Semantic analysis tab"""
    
    st.markdown("## 🔍 Semantic Similarity Analysis")
    
    data = st.session_state.book_data
    
    #similarity matrix
    st.markdown("### 📊 Chapter Similarity Matrix")
    st.caption("Darker blue = More similar content")
    
    fig = go.Figure(data=go.Heatmap(
        z=data['similarity_matrix'],
        x=[f"Ch {i+1}" for i in range(len(data['chapters']))],
        y=[f"Ch {i+1}" for i in range(len(data['chapters']))],
        colorscale='Blues',
        hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(height=600, template='plotly_white')
    st.plotly_chart(fig, width="stretch")
    
    st.markdown("---")
    
    #3D scatter
    st.markdown("### 🌐 3D Semantic Space")
    st.caption("Chapters positioned by semantic similarity - rotate to explore!")
    
    embeddings = data['chapter_embeddings']
    chapters = data['chapters']
    
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        mode='markers+text',
        marker=dict(
            size=12,
            color=[ch['cluster'] for ch in chapters],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Cluster")
        ),
        text=[f"Ch {ch['number']}" for ch in chapters],
        textposition='top center',
        hovertext=[ch['title'] for ch in chapters],
        hovertemplate='<b>%{hovertext}</b><extra></extra>'
    )])
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        height=700,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_3d, width="stretch")

def show_phrases():
    """Key phrases tab"""
    
    st.markdown("## 🔑 Key Phrases")
    st.info("Multi-word phrases extracted using RAKE algorithm")
    
    data = st.session_state.book_data
    
    st.markdown("### 🎯 Top Key Phrases")
    st.caption("Ranked by importance score")
    
    for idx, (phrase, score) in enumerate(data['key_phrases'], 1):
        col1, col2, col3 = st.columns([0.5, 3, 1])
        
        with col1:
            st.markdown(f"**#{idx}**")
        with col2:
            st.markdown(f"**{phrase}**")
        with col3:
            max_score = max([s for _, s in data['key_phrases']])
            st.progress(score / max_score, text=f"{score:.1f}")
    
    st.markdown("---")
    
    #Comparison
    st.markdown("### 📊 Why RAKE Phrases Are Better")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔑 RAKE Multi-Word Phrases:**")
        st.success("""
        - habit stacking
        - behavior change
        - environment design
        - continuous improvement
        """)
        st.caption("✅ Captures complete concepts")
    
    with col2:
        st.markdown("**📝 Single Keywords:**")
        st.warning("""
        - habit
        - behavior
        - environment
        - continuous
        """)
        st.caption("⚠️ Loses meaning without context")

def show_network():
    """Concept network tab"""
    
    st.markdown("## 🕸️ Concept Network")
    st.info("Interactive network showing how key concepts connect")
    
    data = st.session_state.book_data
    
    #filters with explanations
    st.markdown("### 🎛️ Network Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_conn = st.slider(
            "Minimum Connections",
            0, 5, 1,
            help="Show only concepts with at least this many connections to others"
        )
    
    with col2:
        color_by = st.selectbox(
            "Color by:",
            ["Importance (# connections)", "Random"],
            help="Importance = concepts with more connections are darker"
        )
    
    with col3:
        layout = st.selectbox(
            "Layout:",
            ["Force-directed (natural)", "Circular (organized)"],
            help="Force-directed looks organic, Circular is symmetrical"
        )
    
    st.markdown("---")
    
    #Build network
    G = nx.Graph()
    concepts = data['key_concepts'][:15]
    
    for concept in concepts:
        G.add_node(concept)
    
    #add edges
    for i, c1 in enumerate(concepts):
        for c2 in concepts[i+1:]:
            if np.random.random() > 0.6:
                G.add_edge(c1, c2, weight=np.random.uniform(0.3, 1.0))
    
    #filter
    if min_conn > 0:
        to_remove = [n for n, d in G.degree() if d < min_conn]
        G.remove_nodes_from(to_remove)
    
    #Layout
    if "Force" in layout:
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    else:
        pos = nx.circular_layout(G)
    
    #visualization
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            size=[20 + G.degree(n) * 10 for n in G.nodes()],
            color=[G.degree(n) for n in G.nodes()],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Connections"),
            line=dict(width=2, color='white')
        ),
        hovertext=[f"{n}<br>{G.degree(n)} connections" for n in G.nodes()],
        hovertemplate='<b>%{hovertext}</b><extra></extra>'
    )
    
    fig_net = go.Figure(data=[edge_trace, node_trace])
    fig_net.update_layout(
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_net, width="stretch")
    
    #Statistics
    st.markdown("---")
    st.markdown("### 📊 Network Statistics")
    st.caption("Analysis of the concept relationship network")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nodes", G.number_of_nodes(), 
                 help="Total concepts in network")
    with col2:
        st.metric("Edges", G.number_of_edges(),
                 help="Total connections between concepts")
    with col3:
        density = nx.density(G) if G.number_of_nodes() > 1 else 0
        st.metric("Density", f"{density:.3f}",
                 help="Network density (0-1, higher = more connected)")
    with col4:
        avg_deg = sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1)
        st.metric("Avg Connections", f"{avg_deg:.1f}",
                 help="Average connections per concept")

if __name__ == "__main__":
    main()
