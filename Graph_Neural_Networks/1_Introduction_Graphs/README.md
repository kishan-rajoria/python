# Topic 1: Introduction to Graphs & Their Importance

## 1. What are Graphs?

In the context of computer science and machine learning, a graph is a data structure used to represent relationships (edges) between entities (nodes or vertices).

A graph \( G \) is typically defined as a pair \( G = (V, E) \), where:

*   **V** is the set of **nodes** (or vertices).
*   **E** is the set of **edges** (or links) connecting pairs of nodes.

Graphs can model a vast array of real-world systems and abstract structures.

## 2. Why are Graphs Important in ML?

Many real-world datasets are inherently graph-structured, capturing complex relationships that are difficult to model with traditional grid-like structures (like images) or sequences (like text).

Examples include:

*   **Social Networks:** Users are nodes, friendships/follows are edges.
*   **Biological Networks:** Proteins/genes are nodes, interactions are edges.
*   **Knowledge Graphs:** Entities (people, places, concepts) are nodes, relationships (born in, works at, is a type of) are edges.
*   **Molecular Structures:** Atoms are nodes, chemical bonds are edges.
*   **Citation Networks:** Papers are nodes, citations are edges.
*   **Recommender Systems:** Users and items can be nodes, interactions (ratings, purchases) can be edges (forming a bipartite graph).
*   **Transportation Networks:** Locations are nodes, routes are edges.

Traditional ML models (like CNNs for grids, RNNs/Transformers for sequences) often struggle with the irregular structure, variable neighborhood sizes, and permutation invariance of graph data. Graph Neural Networks (GNNs) are specifically designed to operate directly on graph structures.

## 3. Types of Graphs

*   **Directed vs. Undirected:**
    *   *Undirected:* Edges have no direction (e.g., Facebook friendship). (A, B) is the same as (B, A).
    *   *Directed:* Edges have a direction (e.g., Twitter follow, citation). (A, B) is different from (B, A).
*   **Weighted vs. Unweighted:**
    *   *Unweighted:* Edges simply indicate a connection exists.
    *   *Weighted:* Edges have associated weights representing strength, cost, distance, etc.
*   **Homogeneous vs. Heterogeneous:**
    *   *Homogeneous:* All nodes are of the same type, and all edges are of the same type.
    *   *Heterogeneous:* Nodes and/or edges can be of different types (e.g., a graph with User, Product, and Review nodes, and Purchased, Reviewed edges).
*   **Static vs. Dynamic:**
    *   *Static:* Graph structure and features are fixed over time.
    *   *Dynamic:* Graph structure and/or features change over time.
*   **Bipartite Graph:** Nodes can be divided into two disjoint sets such that edges only connect nodes from different sets (e.g., users and items in recommendations).

## 4. Graph Terminology

*   **Node (Vertex):** An entity in the graph.
*   **Edge (Link):** A connection between two nodes.
*   **Neighbor (Adjacent Node):** Nodes directly connected by an edge.
*   **Degree (of a node):** The number of edges connected to a node. In directed graphs, we distinguish between *in-degree* (incoming edges) and *out-degree* (outgoing edges).
*   **Path:** A sequence of nodes connected by edges.
*   **Connected Graph:** An undirected graph where there is a path between any two nodes.
*   **Subgraph:** A graph formed by a subset of nodes and edges from a larger graph.

## 5. Representing Graphs

Common ways to represent graphs computationally for ML:

*   **Adjacency Matrix (A):**
    *   A square matrix where \( A_{ij} = 1 \) if there is an edge from node \( i \) to node \( j \), and 0 otherwise (can store weights instead of 1s for weighted graphs).
    *   Symmetric for undirected graphs.
    *   Can be very sparse and memory-intensive for large graphs (\( |V|^2 \) space).

*   **Edge List:**
    *   A list of pairs (or triplets for weighted graphs) representing the edges.
    *   Example: `[(0, 1), (0, 2), (1, 2), (2, 3)]`
    *   More memory-efficient for sparse graphs.

*   **Adjacency List:**
    *   An array or dictionary where the index/key corresponds to a node, and the value is a list of its neighbors.
    *   Example: `{0: [1, 2], 1: [0, 2], 2: [0, 1, 3], 3: [2]}`
    *   Often a good balance between efficiency for finding neighbors and memory usage for sparse graphs.

*   **Feature Matrix (X):**
    *   A matrix where each row corresponds to a node and contains its features.
    *   Dimensions: \( |V| \times F \), where F is the number of node features.

*   **Edge Feature Matrix (E):** (Less common, but used in some GNNs)
    *   A matrix where each row corresponds to an edge and contains its features.

## 6. Graph-Related Tasks in ML

GNNs can be applied to various tasks:

*   **Node Classification/Regression:** Predict a property of each node (e.g., classify user type, predict protein function).
*   **Graph Classification/Regression:** Predict a property of the entire graph (e.g., classify molecule toxicity, predict graph property).
*   **Link Prediction:** Predict whether an edge exists or will exist between two nodes (e.g., recommend friends, predict protein interactions).
*   **Community Detection/Clustering:** Group nodes into clusters based on graph structure.
*   **Graph Generation:** Create new graphs with desired properties.

## Next Steps

*   **Code Example:** Demonstrate how to represent graphs using libraries like `NetworkX` and potentially visualize a small graph.
*   **Topic 2: Basic Graph Neural Network Concepts** (Message Passing). 