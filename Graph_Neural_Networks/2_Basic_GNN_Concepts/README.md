# Topic 2: Basic Graph Neural Network Concepts (Message Passing)

## 1. Motivation: Why GNNs?

Traditional deep learning models like CNNs and RNNs excel on data with regular structures (grids, sequences). However, graphs have:

*   **Irregular Structure:** Nodes can have arbitrary numbers of neighbors.
*   **No Canonical Node Ordering:** Unlike pixels in an image or words in a sentence, there's no inherent order to nodes in a graph.
*   **Complex Relationships:** Interactions can be long-range and non-local.

Graph Neural Networks (GNNs) are designed to operate directly on graph data, respecting its structure and invariances (e.g., permutation invariance - shuffling node order shouldn't change the output for graph-level tasks).

## 2. The Message Passing Paradigm

Many GNN architectures can be understood through the **message passing** (or neighborhood aggregation) framework. The core idea is that each node updates its feature representation (embedding) by aggregating information from its local neighbors.

A typical message passing layer in a GNN performs the following steps for each node \( v \) in the graph at layer \( k \):

1.  **Message Computation:** For each neighbor \( u \) in the neighborhood \( \mathcal{N}(v) \) of node \( v \), compute a message \( \mathbf{m}_{u \\rightarrow v}^{(k)} \). This message is typically a function of the node features of the neighbor \( u \) (from the previous layer, \( \mathbf{h}_u^{(k-1)} \)) and potentially the edge features \( \mathbf{e}_{uv} \) connecting \( u \) and \( v \).
    \[ \mathbf{m}_{u \\rightarrow v}^{(k)} = \\text{MESSAGE}^{(k)}(\mathbf{h}_u^{(k-1)}, \mathbf{h}_v^{(k-1)}, \mathbf{e}_{uv}) \]

2.  **Aggregation:** Aggregate all incoming messages for node \( v \) from its neighbors into a single vector \( \mathbf{m}_v^{(k)} \). Common aggregation functions (\( \text{AGGREGATE} \)) include:
    *   Sum
    *   Mean
    *   Max
    *   (Learnable aggregators like Attention)
    \[ \mathbf{m}_v^{(k)} = \\text{AGGREGATE}^{(k)}(\\{ \mathbf{m}_{u \\rightarrow v}^{(k)} : u \\in \\mathcal{N}(v) \\}) \]
    The aggregation function needs to be permutation invariant (order of neighbors doesn't matter).

3.  **Update:** Combine the aggregated message \( \mathbf{m}_v^{(k)} \) with the node's own previous representation \( \mathbf{h}_v^{(k-1)} \) to compute the node's new representation \( \mathbf{h}_v^{(k)} \) for the current layer.
    \[ \mathbf{h}_v^{(k)} = \\text{UPDATE}^{(k)}(\mathbf{h}_v^{(k-1)}, \mathbf{m}_v^{(k)}) \]

The \( \text{MESSAGE} \), \( \text{AGGREGATE} \), and \( \text{UPDATE} \) functions are typically parameterized (e.g., using learnable weight matrices, often shared across all nodes) and incorporate non-linear activation functions (like ReLU). These parameters are learned during training via backpropagation based on the specific task loss.

## 3. Stacking Layers

Multiple message passing layers are usually stacked. Each layer effectively expands the "receptive field" of a node, allowing it to incorporate information from neighbors that are further away.

*   After 1 layer, node \( v \) incorporates information from its 1-hop neighbors.
*   After 2 layers, node \( v \) incorporates information from its 2-hop neighbors.
*   After \( K \) layers, node \( v \) incorporates information from its K-hop neighbors.

The final node embeddings \( \mathbf{h}_v^{(K)} \) after \( K \) layers can then be used for downstream tasks (node classification, link prediction, etc.).

## 4. Example: Graph Convolutional Network (GCN)

A simplified GCN layer (Kipf & Welling, 2017) update rule can be seen as a specific instance of message passing:

\[ \mathbf{H}^{(k)} = \\sigma \left( \\tilde{\mathbf{D}}^{-rac{1}{2}} \\tilde{\mathbf{A}} \\tilde{\mathbf{D}}^{-rac{1}{2}} \\mathbf{H}^{(k-1)} \\mathbf{W}^{(k)} \right) \]

Where:
*   \( \mathbf{H}^{(k)} \) is the matrix of node embeddings at layer \( k \).
*   \( \mathbf{A} \) is the adjacency matrix.
*   \( \mathbf{I} \) is the identity matrix.
*   \( \tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I} \) (adjacency matrix with self-loops added).
*   \( \tilde{\mathbf{D}} \) is the diagonal degree matrix of \( \tilde{\mathbf{A}} \) (\( \tilde{D}_{ii} = \\sum_j \\tilde{A}_{ij} \)).
*   \( \mathbf{W}^{(k)} \) is the learnable weight matrix for layer \( k \).
*   \( \sigma \) is a non-linear activation function (e.g., ReLU).

The term \( \tilde{\mathbf{D}}^{-rac{1}{2}} \\tilde{\mathbf{A}} \\tilde{\mathbf{D}}^{-rac{1}{2}} \) is a form of symmetrically normalized adjacency matrix. Multiplying by \( \mathbf{H}^{(k-1)} \) performs the aggregation (a weighted sum/mean of neighbor features and self features), and multiplying by \( \mathbf{W}^{(k)} \) followed by \( \sigma \) performs the update/transformation.

## 5. Key Considerations

*   **Choice of Aggregator:** Different aggregators (sum, mean, max) have different properties and may be suited for different tasks.
*   **Depth:** Deeper GNNs can capture information from larger neighborhoods but risk *over-smoothing* (node embeddings becoming too similar) and are harder to train.
*   **Input Features:** The quality of initial node features \( \mathbf{H}^{(0)} \) significantly impacts performance. If no features are available, one might use node degrees or constant values.

## Next Steps

*   **Code Example:** Implement a conceptual message passing step manually on a small graph, or show the structure of a basic GNN layer using a library like PyTorch Geometric or DGL.
*   **Topic 3: Common GNN Layer Types** (GCN, GraphSAGE, GAT). 