# Ensure required libraries are installed:
# pip install networkx matplotlib numpy pandas

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("--- GNNs: Introduction to Graph Representation with NetworkX ---")

# --- 1. Creating a Graph ---
# We can create an empty graph (undirected by default)
G = nx.Graph()

# Or create a directed graph
# G_directed = nx.DiGraph()

print(f"Created an empty graph: {G}")

# --- 2. Adding Nodes ---
# Add nodes one by one
G.add_node(0)
G.add_node(1)
G.add_node("NodeC") # Nodes can be any hashable object

# Add nodes from a list
G.add_nodes_from([3, 4])

# Add nodes with attributes (features)
G.add_node(5, feature1="valueA", feature2=10)
G.add_node(6, feature1="valueB", feature2=20)

print(f"\nNodes in the graph: {G.nodes()}")
print(f"Attributes of node 5: {G.nodes[5]}")

# --- 3. Adding Edges ---
# Add edges one by one (connecting existing or new nodes)
G.add_edge(0, 1)
G.add_edge("NodeC", 3)

# Add edges from a list
G.add_edges_from([(1, 3), (3, 4), (4, 5)])

# Add edges with attributes (weights or features)
G.add_edge(5, 6, weight=2.5, relation="friend")
G.add_edge(0, 4, weight=1.0)

print(f"\nEdges in the graph: {G.edges()}")
print(f"Attributes of edge (5, 6): {G.edges[5, 6]}")

# --- 4. Accessing Graph Properties ---
print(f"\nNumber of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Neighbors of a node
node_to_check = 3
print(f"Neighbors of node {node_to_check}: {list(G.neighbors(node_to_check))}")

# Degree of a node
print(f"Degree of node {node_to_check}: {G.degree(node_to_check)}")

# --- 5. Graph Representations --- 
print("\n--- Common Graph Representations ---")

# a) Adjacency Matrix (as a NumPy array or SciPy sparse matrix)
# Note: Requires nodes to be consistently ordered (usually 0, 1, 2,...)
# We'll use a subgraph with integer nodes for simplicity here
subgraph_nodes = [0, 1, 3, 4]
subgraph = G.subgraph(subgraph_nodes)
adj_matrix = nx.to_numpy_array(subgraph, nodelist=sorted(subgraph.nodes()))
print("Adjacency Matrix (for subgraph [0, 1, 3, 4]):")
print(adj_matrix)

# b) Edge List
print(f"\nEdge List: {list(G.edges())}")

# c) Adjacency List
print("\nAdjacency List (Dictionary format):")
adj_list = {node: list(neighbors) for node, neighbors in G.adjacency()}
print(adj_list)

# --- 6. Node Features (Conceptual for GNNs) ---
# In GNNs, we often have a feature matrix X where each row is a node's feature vector.
# Let's create a dummy one based on the attributes we added.

node_features = []
feature_map = {}
node_list_ordered = sorted(G.nodes(), key=lambda x: str(x)) # Ensure consistent order

for i, node in enumerate(node_list_ordered):
    attrs = G.nodes[node]
    # Simple encoding: feature2 if present, else use node index as dummy feature
    feat2 = attrs.get('feature2', i) # Use index i as dummy if feature2 is missing
    node_features.append([feat2]) # Example: using only feature2
    feature_map[node] = i # Map node ID to row index in feature matrix

X = np.array(node_features)
print("\nNode Feature Matrix (X - Conceptual Example):")
print(f"(Node order for rows: {node_list_ordered})")
print(X)

# --- 7. Visualizing the Graph ---
print("\n--- Visualizing the Graph --- ")

plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')

# Draw edges
nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.5)
edge_weights = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12)

plt.title("Simple Graph Visualization")
plt.axis('off') # Turn off the axis
# plt.show() # Uncomment to display the plot interactively
print("Graph visualization generated (plot window might open if plt.show() is uncommented).")

# Saving the plot (optional)
try:
    plt.savefig("graph_visualization.png")
    print("Graph visualization saved to graph_visualization.png")
except Exception as e:
    print(f"Could not save graph visualization: {e}")


print("\n--- Script Finished --- ")
print("This script demonstrated basic graph creation, manipulation, representation,")
print("and visualization using NetworkX.") 