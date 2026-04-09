# Topic 3: Common GNN Layer Types

Building upon the message passing framework, several specific GNN layer types have been developed, each with different ways of defining the MESSAGE, AGGREGATE, and UPDATE functions. Here are some of the most common ones:

## 1. Graph Convolutional Network (GCN)

*   **Reference:** Kipf & Welling (2017)
*   **Core Idea:** Performs a localized spectral convolution using a simplified, efficient approximation. It aggregates features from the node itself and its immediate neighbors.
*   **Simplified Update Rule (Matrix Form):**
    \[ \mathbf{H}^{(k)} = \sigma \left( \hat{\mathbf{D}}^{-1/2} \hat{\mathbf{A}} \hat{\mathbf{D}}^{-1/2} \mathbf{H}^{(k-1)} \mathbf{W}^{(k)} \right) \]
    Where \( \hat{\mathbf{A}} = \mathbf{A} + \mathbf{I} \) (adjacency matrix with self-loops) and \( \hat{\mathbf{D}} \) is the diagonal degree matrix of \( \hat{\mathbf{A}} \).
*   **Message Passing View:**
    *   Aggregation is effectively a **normalized mean** of the transformed features of the neighbors (including the node itself).
    *   \( \text{AGGREGATE}(\{\mathbf{h}_u^{(k-1)} : u \in \mathcal{N}(v) \cup \{v\}\}) = \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\hat{d}_v \hat{d}_u}} \mathbf{h}_u^{(k-1)} \)
    *   Update involves a linear transformation with \( \mathbf{W}^{(k)} \) and an activation \( \sigma \).
*   **Pros:** Simple, computationally efficient, often a strong baseline.
*   **Cons:** Aggregation is fixed (mean); may struggle with varying neighbor importance; typically used for transductive learning (requires the full graph during training).
*   **PyTorch Geometric:** `torch_geometric.nn.GCNConv`

## 2. GraphSAGE (Sample and Aggregate)

*   **Reference:** Hamilton, Ying, & Leskovec (2017)
*   **Core Idea:** Designed for inductive learning (generalizing to unseen nodes/graphs). Instead of using all neighbors, it **samples** a fixed number of neighbors for each node at each layer and then applies an aggregation function.
*   **Update Rule (Simplified Node `v`):**
    1.  Sample a neighborhood \( \mathcal{N}_S(v) \subseteq \mathcal{N}(v) \).
    2.  Aggregate features from sampled neighbors: \( \mathbf{h}_{\mathcal{N}_S(v)}^{(k)} = \text{AGGREGATE}(\{\mathbf{h}_u^{(k-1)}, \forall u \in \mathcal{N}_S(v)\}) \)
    3.  Update node \( v \)'s embedding: \( \mathbf{h}_v^{(k)} = \sigma \left( \mathbf{W}^{(k)} \cdot \text{CONCAT}(\mathbf{h}_v^{(k-1)}, \mathbf{h}_{\mathcal{N}_S(v)}^{(k)}) \right) \)
*   **Aggregation Functions:** Can use various aggregators:
    *   **Mean Aggregator:** Average of neighbor features (similar to GCN but without the specific normalization).
    *   **Pooling Aggregator (Max/Mean Pooling):** Apply a shared MLP to neighbor features, then apply element-wise max or mean pooling.
    *   **LSTM Aggregator:** Apply LSTM to a random permutation of neighbor features for sequence-like aggregation.
*   **Pros:** Inductive capability, scalable due to neighbor sampling, flexible aggregation.
*   **Cons:** Sampling adds randomness; performance might depend on sampling strategy.
*   **PyTorch Geometric:** `torch_geometric.nn.SAGEConv`

## 3. Graph Attention Network (GAT)

*   **Reference:** Veličković et al. (2018)
*   **Core Idea:** Addresses the limitation of GCN and GraphSAGE having fixed aggregation weights. GAT uses **masked self-attention** to assign different importance scores (attention weights) to different neighbors during aggregation.
*   **Mechanism:**
    1.  Apply a linear transformation (with weight matrix \( \mathbf{W} \)) to node features: \( \mathbf{z}_i = \mathbf{W} \mathbf{h}_i \).
    2.  Compute attention coefficients \( e_{ij} \) between node \( i \) and its neighbor \( j \) using a shared attention mechanism \( a \) (e.g., a single-layer feedforward network):
        \[ e_{ij} = a(\mathbf{z}_i, \mathbf{z}_j) \]
    3.  Normalize coefficients using softmax across all neighbors \( j \) of node \( i \) to get attention weights \( \alpha_{ij} \):
        \[ \alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})} \]
    4.  Aggregate neighbor features using a weighted sum based on attention weights:
        \[ \mathbf{h}'_i = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{z}_j \right) \]
    5.  Often uses **multi-head attention** (running independent attention mechanisms in parallel and concatenating/averaging results) to stabilize learning and capture different aspects of relationships.
*   **Pros:** Assigns importance weights to neighbors (interpretable), often achieves state-of-the-art results, works inductively.
*   **Cons:** Computationally more expensive than GCN or GraphSAGE due to attention calculation.
*   **PyTorch Geometric:** `torch_geometric.nn.GATConv`

## 4. Other Notable Layers

*   **Graph Isomorphism Network (GIN):** Theoretically shown to be maximally powerful among message-passing GNNs under certain conditions. Uses MLPs and sum aggregation. (`torch_geometric.nn.GINConv`)
*   **EdgeConv:** Operates on edge features or dynamically computes them based on connected node features, suitable for point clouds and dynamic graphs. (`torch_geometric.nn.EdgeConv`)
*   **ChebConv:** Uses Chebyshev polynomial approximations for graph convolutions, related to spectral methods. (`torch_geometric.nn.ChebConv`)

## Next Steps

*   **Code Example:** Show how to define GNN models using different PyG layers (`GCNConv`, `SAGEConv`, `GATConv`) and compare their structure.
*   **Topic 4: Building a Complete GNN Model (Node Classification Example)** 