# Ensure required libraries are installed:
# pip install torch torch_geometric
# (May require specific PyTorch version depending on your CUDA setup, see PyG installation instructions)

import torch
import torch.nn.functional as F
import torch.optim as optim

# Try importing PyTorch Geometric components
try:
    from torch_geometric.datasets import Planetoid # For Cora, Citeseer, Pubmed
    from torch_geometric.nn import GCNConv
    import torch_geometric.transforms as T # For data transformations
except ImportError:
    print("PyTorch Geometric not found or dataset dependencies missing.")
    print("Please install it following the instructions at: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    exit()

print(f"--- GNNs: Node Classification Example (Cora Dataset) --- ")
print(f"Using PyTorch version: {torch.__version__}")

# --- 1. Load Cora Dataset ---
# Planetoid datasets automatically download if not present.
# transform=T.NormalizeFeatures() applies row-normalization to node features.
dataset_name = 'Cora'
path = './data/Planetoid' # Directory to store the dataset
dataset = Planetoid(root=path, name=dataset_name, transform=T.NormalizeFeatures())

data = dataset[0] # Get the single graph object.

print("\n--- Dataset Information --- ")
print(f'Dataset: {dataset_name}')
print(f'  Number of graphs: {len(dataset)}')
print(f'  Number of nodes: {data.num_nodes}')
print(f'  Number of edges: {data.num_edges}')
print(f'  Number of features per node: {dataset.num_node_features}')
print(f'  Number of classes: {dataset.num_classes}')
print(f'  Has isolated nodes: {data.has_isolated_nodes()}')
print(f'  Has self-loops: {data.has_self_loops()}')
print(f'  Is undirected: {data.is_undirected()}')
print(f'  Node labels y shape: {data.y.shape}')
print(f'  Train mask sum: {data.train_mask.sum().item()}') # Number of training nodes
print(f'  Validation mask sum: {data.val_mask.sum().item()}') # Number of validation nodes
print(f'  Test mask sum: {data.test_mask.sum().item()}')   # Number of test nodes

# --- 2. Define the GCN Model --- 
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout_p = 0.5 # Dropout probability

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv2(x, edge_index)

        # Return log-probabilities for NLLLoss
        return F.log_softmax(x, dim=1)

# --- 3. Instantiate Model and Optimizer --- 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')

model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
data = data.to(device) # Move data to the device

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("\n--- GCN Model Structure --- ")
print(model)

# --- 4. Define Training and Evaluation Functions --- 

def train():
    model.train() # Set model to training mode (enables dropout)
    optimizer.zero_grad() # Clear gradients
    out = model(data) # Forward pass
    # Calculate loss only on training nodes using the train_mask
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward() # Backward pass (compute gradients)
    optimizer.step() # Update weights
    return loss.item()

@torch.no_grad() # Decorator to disable gradient calculation for evaluation
def test():
    model.eval() # Set model to evaluation mode (disables dropout)
    out = model(data) # Forward pass
    pred = out.argmax(dim=1) # Get predicted classes

    accuracies = {}
    for mask_name, mask in [('Train', data.train_mask), ('Val', data.val_mask), ('Test', data.test_mask)]:
        correct = (pred[mask] == data.y[mask]).sum()
        acc = int(correct) / int(mask.sum())
        accuracies[mask_name] = acc
    return accuracies

# --- 5. Training Loop --- 
print("\n--- Starting Training --- ")
num_epochs = 200

for epoch in range(1, num_epochs + 1):
    loss = train()
    if epoch % 10 == 0:
        accuracies = test()
        train_acc = accuracies['Train']
        val_acc = accuracies['Val']
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

print("--- Training Finished --- ")

# --- 6. Final Evaluation --- 
final_accuracies = test()
print("\n--- Final Evaluation --- ")
print(f"Final Train Accuracy: {final_accuracies['Train']:.4f}")
print(f"Final Validation Accuracy: {final_accuracies['Val']:.4f}")
print(f"Final Test Accuracy: {final_accuracies['Test']:.4f}")

print("\n--- Script Finished --- ") 