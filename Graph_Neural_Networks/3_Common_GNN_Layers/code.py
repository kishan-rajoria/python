# Ensure required libraries are installed:
# pip install torch torch_geometric
# (May require specific PyTorch version depending on your CUDA setup, see PyG installation instructions)

import torch
import torch.nn.functional as F

# Try importing PyTorch Geometric components
try:
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data
except ImportError:
    print("PyTorch Geometric not found.")
    print("Please install it following the instructions at: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    exit()

print(f"--- GNNs: Common Layer Types (PyTorch Geometric) --- ")
print(f"Using PyTorch version: {torch.__version__}")

# --- 1. Sample Graph Data (Reusing from Topic 2) ---
num_nodes = 4
num_node_features = 2
x = torch.tensor([
    [-1, 1], [0, -1], [1, 0], [2, 2]
], dtype=torch.float)
edge_index = torch.tensor([
    [0, 0, 1, 2, 1, 2, 3],  # Source nodes
    [1, 2, 2, 3, 0, 0, 2]   # Target nodes
], dtype=torch.long)
graph_data = Data(x=x, edge_index=edge_index)

print("\n--- Sample Graph Data --- ")
print(graph_data)

# --- 2. Define Models with Different Layers --- 

hidden_channels = 8 # Intermediate feature dimension
output_channels = 4 # Final embedding dimension

# Model 1: Using GCNConv
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training) # Dropout usually applied during training
        x = self.conv2(x, edge_index)
        return x

# Model 2: Using SAGEConv (GraphSAGE)
class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv allows specifying the aggregation method (e.g., 'mean', 'max', 'lstm')
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr='mean')

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Model 3: Using GATConv (Graph Attention Network)
class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super().__init__()
        # GATConv uses attention. `heads` specifies the number of attention heads.
        # The output dimension will be heads * hidden_channels for the first layer.
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        # Adjust input channels for the second layer if using multi-head attention.
        # The final layer often uses heads=1 or averages the head outputs.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Dropout is often applied to attention weights internally in GATConv during training
        x = F.dropout(x, p=0.6, training=self.training) # Input feature dropout
        x = self.conv1(x, edge_index)
        x = F.elu(x) # ELU activation is common with GAT
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # Output is typically class scores (e.g., log_softmax) for classification
        return x # F.log_softmax(x, dim=1)

# --- 3. Instantiate and Test Models (Forward Pass Only) ---

print("\n--- Instantiating Models ---")

# Instantiate GCN Model
model_gcn = GCNModel(num_node_features, hidden_channels, output_channels)
print("GCN Model Structure:")
print(model_gcn)

# Instantiate GraphSAGE Model
model_sage = GraphSAGEModel(num_node_features, hidden_channels, output_channels)
print("\nGraphSAGE Model Structure:")
print(model_sage)

# Instantiate GAT Model
model_gat = GATModel(num_node_features, hidden_channels, output_channels, heads=2)
print("\nGAT Model Structure:")
print(model_gat)

print("\n--- Performing Forward Pass (No Training) ---")

# Set models to evaluation mode
model_gcn.eval()
model_sage.eval()
model_gat.eval()

with torch.no_grad():
    out_gcn = model_gcn(graph_data)
    out_sage = model_sage(graph_data)
    out_gat = model_gat(graph_data)

print(f"\nGCN Output Shape: {out_gcn.shape}")
# print("GCN Output Sample:\n", out_gcn)

print(f"\nGraphSAGE Output Shape: {out_sage.shape}")
# print("GraphSAGE Output Sample:\n", out_sage)

print(f"\nGAT Output Shape: {out_gat.shape}")
# print("GAT Output Sample:\n", out_gat)

print("\nObservations:")
print(" - All models produce node embeddings of the desired output shape.")
print(" - The internal computations differ based on the layer type (GCN normalization, SAGE aggregation, GAT attention)." )

print("\n--- Script Finished --- ")
print("This script showed how to define simple GNNs with different common layer types in PyG.") 