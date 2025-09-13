import networkx as nx
from pyvis.network import Network

# --- Step 1: Define concepts + descriptions ---
descriptions = {
    "cue": "Trigger that initiates the habit loop.",
    "craving": "Motivational urge that follows the cue.",
    "response": "The action you take.",
    "reward": "Benefit that reinforces the habit.",
    "habit": "A routine behavior formed by repetition.",
    "identity": "Self-image shaped by repeated actions.",
    "belief": "Convictions that influence habits.",
    "environment": "Context that makes habits easier or harder.",
    "behavior": "Observable actions stemming from cues/cravings.",
    "routine": "Repeated sequence inside a habit.",
    "system": "Structures that make habits sustainable."
}

# --- Step 2: Build NetworkX Graph ---
G = nx.Graph()

# add nodes
for node in descriptions:
    G.add_node(node)

# add edges (habit loop + extra connections)
edges = [
    ("cue", "craving"),
    ("craving", "response"),
    ("response", "reward"),
    ("reward", "habit"),
    ("habit", "identity"),
    ("identity", "belief"),
    ("habit", "behavior"),
    ("habit", "routine"),
    ("habit", "system"),
    ("habit", "environment"),
    ("environment", "cue"),
    ("system", "routine"),
]
G.add_edges_from(edges)

# --- Step 3: PyVis Visualization ---
net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False)

# Use ForceAtlas2 for spread-out layout
net.force_atlas_2based(
    gravity=-25,
    central_gravity=0.01,
    spring_length=180,
    spring_strength=0.02,
    damping=0.35,
    overlap=1
)

# Add nodes with hover descriptions
for node in G.nodes():
    net.add_node(
        node,
        label=node,
        title=descriptions.get(node, "No description available"),
        size=28 if node in ["cue","craving","response","reward","habit","identity"] else 20,
        color="#E63946" if node=="habit" else "#457B9D" if node in ["identity","belief"] else "#FF8C00" if node in ["cue","craving","response","reward"] else "#A8DADC"
    )

# Add edges
for u, v in G.edges():
    net.add_edge(u, v)

# Optional: UI controls in the HTML
net.show_buttons(filter_=["physics"])

# Save graph
net.save_graph("graph/atomic_habits_concept_network_clean.html")
print("✅ Graph saved to graph/atomic_habits_concept_network_clean.html")
