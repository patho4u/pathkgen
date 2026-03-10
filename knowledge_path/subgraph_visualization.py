import pandas as pd
import networkx as nx
import gravis as gv
import os
import sys

# Import base directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths_config import DATA_DIR

df = pd.read_csv(os.path.join(DATA_DIR, "wsi_bench", "custom_kg", "relations.csv"))

G = nx.Graph()

# Build Graph
for _, row in df.iterrows():
    src = row["reference_entity"]
    tgt = row["related_entity"]
    rel = row["relation_name"]
    G.add_edge(src, tgt, label=rel)

center_node = "Head and Neck Squamous Cell Carcinoma"

# Nodes to highlight
green_nodes = {"Keratinization", "intercellular bridge", "Sheet-Like Growth Pattern"}
red_nodes = {"Lymph-vascular invasion", "Perineural Invasion", "Necrosis"}

# Default styling (everything grey first)
for n in G.nodes():
    G.nodes[n]["color"] = "#b5b3b3"
    G.nodes[n]["size"] = 15

for u, v in G.edges():
    G.edges[u, v]["color"] = "#cccccc"
    G.edges[u, v]["width"] = 2

# Highlight center node
if center_node in G.nodes():
    G.nodes[center_node]["color"] = "#00008b"  # dark blue
    G.nodes[center_node]["size"] = 15

# Highlight green nodes
for node in green_nodes:
    if node in G.nodes():
        G.nodes[node]["color"] = "#2ca02c"  # green
        G.nodes[node]["size"] = 15
        # Only highlight edge if it exists
        if G.has_edge(center_node, node):
            G.edges[center_node, node]["color"] = "#2ca02c"
            G.edges[center_node, node]["width"] = 6

# Highlight red nodes (no edges to center)
for node in red_nodes:
    if node in G.nodes():
        G.nodes[node]["color"] = "#d62728"  # red
        G.nodes[node]["size"] = 15
        if G.has_edge(center_node, node):
            G.edges[center_node, node]["color"] = "#d62728"
            G.edges[center_node, node]["width"] = 6

# Draw graph
fig = gv.d3(
    G,
    graph_height=900,
    edge_label_data_source="label",
    many_body_force_strength=-50,
    links_force_distance=35,
    links_force_strength=0.6,
    collision_force_radius=8
)

fig.display()