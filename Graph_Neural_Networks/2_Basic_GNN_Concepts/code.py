# Ensure required libraries are installed:
# pip install torch torch_geometric
# (May require specific PyTorch version depending on your CUDA setup, see PyG installation instructions)

import torch
import torch.nn.functional as F

# Try importing PyTorch Geometric components
try:
    from torch_geometric.nn import GCNConv # Graph Convolutional Network Layer
    from torch_geometric.data import Data
except ImportError:
    print("PyTorch Geometric not found.")
    print("Please install it following the instructions at: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    exit()

print(f"--- GNNs: Basic GNN Layer Structure (PyTorch Geometric) --- ")
print(f"Using PyTorch version: {torch.__version__}")

# --- 1. Sample Graph Data Representation (PyG Format) ---
# Let's define a simple graph with 4 nodes and some edges.
# Nodes: 0, 1, 2, 3
# Edges: (0, 1), (0, 2), (1, 2), (2, 3)

# a) Node Features (X): Each node has a feature vector.
# Let's assume 2 features per node.
num_nodes = 4
num_node_features = 2
x = torch.tensor([
    [-1, 1],  # Features for Node 0
    [0, -1], # Features for Node 1
    [1, 0],  # Features for Node 2
    [2, 2]   # Features for Node 3
], dtype=torch.float)

# b) Edge Index (edge_index): Defines the graph connectivity.
# Represented as a tensor of shape [2, num_edges].
# Each column [u, v] represents a directed edge from node u to node v.
# For undirected graphs, add edges in both directions.
edge_index = torch.tensor([
    [0, 0, 1, 2, 1, 2, 3],  # Source nodes
    [1, 2, 2, 3, 0, 0, 2]   # Target nodes
], dtype=torch.long)
# Edges: (0->1), (0->2), (1->2), (2->3)
# Make it undirected by adding reverse edges: (1->0), (2->0), (2->1), (3->2)

# c) Create a PyG Data object
# This object bundles all graph information together.
graph_data = Data(x=x, edge_index=edge_index)

print("\n--- Sample Graph Data (PyG format) ---")
print(graph_data)
print(f"Number of nodes: {graph_data.num_nodes}")
print(f"Number of edges: {graph_data.num_edges}")
print(f"Node features shape: {graph_data.x.shape}")
print(f"Edge index shape: {graph_data.edge_index.shape}")

# --- 2. Define a Simple GNN Model --- 
# This model contains a single GCN layer.

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_output_features):
        super().__init__()
        # Define the GCN layer
        # Takes input node features, outputs features of size num_output_features
        self.conv1 = GCNConv(num_node_features, num_output_features)

    def forward(self, data):
        # Extract node features and edge connectivity from the Data object
        x, edge_index = data.x, data.edge_index

        # Apply the GCN layer
        # This performs the message passing (aggregation & update)
        x = self.conv1(x, edge_index)

        # Apply an activation function (optional, depends on the task)
        x = F.relu(x)

        # Apply dropout (optional, for training)
        # x = F.dropout(x, training=self.training)

        # In a multi-layer GNN, you would pass the output x to the next layer.
        # For a node classification task, you might add a final linear layer or softmax.

        return x

# --- 3. Instantiate the Model and Perform Forward Pass ---

# Define the desired output feature dimension for the GNN layer
output_feature_dim = 4

# Instantiate the model
model = SimpleGCN(num_node_features=num_node_features, num_output_features=output_feature_dim)
print("\n--- Simple GNN Model Structure ---")
print(model)

# Perform a forward pass (conceptual, no training involved)
# Set model to evaluation mode (important if dropout/batchnorm were used)
model.eval()
with torch.no_grad(): # Deactivate autograd for inference
    # Pass the graph data through the model
    output_embeddings = model(graph_data)

print("\n--- Output Node Embeddings (After GCN layer) ---")
print(f"Shape: {output_embeddings.shape}") # Should be [num_nodes, output_feature_dim]
print(output_embeddings)

print("\nObservations:")
print(" - The GCN layer transformed the input node features (shape [4, 2])")
print(f"   into output node embeddings (shape [{num_nodes}, {output_feature_dim}])." )
print(" - Each output row represents the updated feature vector for a node,")
print("   incorporating information from its neighbors via message passing.")

print("\n--- Script Finished --- ")
print("This script showed the basic structure of a GNN layer using PyTorch Geometric.")
print("It demonstrated data representation and a single forward pass.") 