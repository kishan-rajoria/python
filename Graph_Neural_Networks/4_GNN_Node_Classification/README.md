# Topic 4: Building a Complete GNN Model (Node Classification Example)

This topic demonstrates how to build, train, and evaluate a Graph Neural Network for a common graph learning task: semi-supervised node classification.

## 1. Task Definition: Semi-Supervised Node Classification

Given a graph where only a subset of nodes have labels (e.g., categories, classes), the goal is to predict the labels for the unlabeled nodes.

*   **Input:** A graph \( G = (V, E) \), a node feature matrix \( X \), and labels \( Y_L \) for a subset of nodes \( V_L \subset V \).
*   **Output:** Predicted labels \( \hat{Y}_U \) for the unlabeled nodes \( V_U = V \setminus V_L \).
*   **Semi-Supervised:** We use both the node features, the graph structure (connectivity), and the known labels to make predictions.

## 2. Standard Datasets

Common benchmark datasets for this task include citation networks:

*   **Cora:** Scientific publications classified into subjects. Nodes are papers, edges are citations. Node features are bag-of-words vectors.
*   **Citeseer:** Similar to Cora.
*   **Pubmed:** Biomedical publications.

These datasets are readily available in libraries like PyTorch Geometric (`torch_geometric.datasets`).

## 3. Steps to Build and Train a GNN Model

Building a GNN for node classification typically involves these steps:

1.  **Load Data:** Load the graph dataset (including node features, edge index, labels, and train/validation/test masks).
2.  **Define Model:** Create a GNN model class (using `torch.nn.Module`) stacking GNN layers (e.g., `GCNConv`, `SAGEConv`, `GATConv`) and potentially dropout layers. The final layer should output scores for each class (dimension = number of classes).
3.  **Choose Loss Function:** For multi-class classification, **Cross-Entropy Loss** (`torch.nn.CrossEntropyLoss` or `F.nll_loss` with `F.log_softmax`) is typically used.
4.  **Choose Optimizer:** Select an optimizer like **Adam** (`torch.optim.Adam`) to update the model weights during training.
5.  **Training Loop:**
    *   Iterate for a specified number of epochs.
    *   **Forward Pass:** Feed the graph data through the model to get output logits/probabilities.
    *   **Calculate Loss:** Compute the loss between the model output and the true labels for the **training nodes** (using the train mask).
    *   **Backward Pass:** Compute gradients of the loss with respect to model parameters (`loss.backward()`).
    *   **Optimizer Step:** Update model parameters (`optimizer.step()`).
    *   **Zero Gradients:** Clear gradients for the next iteration (`optimizer.zero_grad()`).
    *   (Optional) Evaluate on the validation set periodically to monitor performance and potentially implement early stopping.
6.  **Evaluation:**
    *   After training, switch the model to evaluation mode (`model.eval()`).
    *   Perform a forward pass on the graph data.
    *   Calculate the accuracy (or other relevant metrics) on the **test nodes** (using the test mask).

## 4. Implementation Details

*   **Masks:** Datasets like Cora often provide boolean masks (`train_mask`, `val_mask`, `test_mask`) indicating which nodes belong to the training, validation, and test sets respectively. The loss is calculated only on the training nodes, while evaluation uses the validation and test nodes.
*   **Full-Graph Training:** In many standard node classification setups (transductive learning), the model sees the entire graph structure and all node features during training, but only the labels of the training nodes are used for computing the loss.
*   **Output Layer:** The final output of the GNN for classification is usually passed through a `LogSoftmax` function, and the loss is calculated using Negative Log-Likelihood Loss (`F.nll_loss`).

## Next Steps

*   **Code Example:** Implement a complete node classification pipeline on the Cora dataset using PyTorch Geometric, including data loading, model definition (e.g., a 2-layer GCN), training loop, and evaluation. 