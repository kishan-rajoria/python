# Topic 8: Transfer Learning & Fine-tuning for Computer Vision

## Overview

Training state-of-the-art deep learning models (like ResNet, Inception, ViT) from scratch requires massive labeled datasets (e.g., ImageNet with over a million images and 1000 classes) and significant computational resources (GPU time). For many practical computer vision tasks, collecting such large datasets is infeasible.

**Transfer Learning** provides a powerful solution. It leverages knowledge learned by a model pre-trained on a large, general dataset (like ImageNet) and adapts it to a new, often smaller, target dataset for a specific task.

## 1. Why Transfer Learning Works for Vision

CNNs trained on large datasets like ImageNet learn hierarchical features:

*   **Early Layers:** Detect general, low-level features like edges, corners, colors, and textures.
*   **Mid Layers:** Combine low-level features to detect more complex motifs and parts of objects.
*   **Later Layers:** Detect high-level object-specific features and eventually combine them for classification.

The key insight is that the low-level and mid-level features learned on a general dataset (like ImageNet) are often useful for many other visual tasks, even if the specific object classes are different. Transfer learning reuses these learned feature extraction capabilities.

## 2. Strategies for Transfer Learning

The approach depends on the size and similarity of your target dataset compared to the original dataset the model was pre-trained on (e.g., ImageNet).

### Strategy A: Use as a Fixed Feature Extractor

*   **When to use:** Your target dataset is small, and its task is similar to the pre-training task (e.g., classifying different types of flowers using an ImageNet model).
*   **How:**
    1.  Load a pre-trained CNN model (e.g., ResNet-50, VGG-16) without its final classification layer (the "head").
    2.  Freeze the weights of all the convolutional layers (prevent them from being updated during training).
    3.  Pass your target dataset images through the frozen convolutional base to extract feature vectors.
    4.  Train a new, smaller classifier (e.g., a linear SVM, a small fully connected neural network) on these extracted features and your target labels.
*   **Pros:** Fast, computationally inexpensive (only training the small classifier). Less prone to overfitting on small datasets.
*   **Cons:** May not be optimal if the target task requires slightly different features than those learned during pre-training.

### Strategy B: Fine-tuning the Pre-trained Model

*   **When to use:** Your target dataset is reasonably large (thousands of images) or somewhat different from the pre-training dataset.
*   **How:**
    1.  Load a pre-trained CNN model *with* its weights.
    2.  Replace the final classification layer (head) with a new one suited to your target task (e.g., with the number of output neurons matching your number of classes).
    3.  **Fine-tuning:** Continue training the *entire* network (or parts of it) on your new dataset, typically using a very **low learning rate**.
*   **Fine-tuning Variations:**
    *   **Train Only the Head:** Freeze all convolutional layers and train only the newly added classification head. Similar to Strategy A but within the same network structure. Good starting point for small datasets.
    *   **Train Head and Top Layers:** Freeze the early convolutional layers (which learned general features) and unfreeze/train the later convolutional layers *and* the new head. Allows the model to adapt the higher-level features to the new dataset. Good for medium-sized datasets or datasets somewhat different from the pre-training one.
    *   **Train Entire Network:** Unfreeze all layers and train the entire network with a low learning rate. Best approach if you have a large target dataset, allowing maximum adaptation.
*   **Low Learning Rate:** Crucial for fine-tuning. Since the pre-trained weights are already good, you only want to make small adjustments. Using a large learning rate could destroy the learned features.
*   **Pros:** Can achieve higher accuracy than fixed feature extraction by adapting features to the new task.
*   **Cons:** Requires more computation and data than Strategy A. Higher risk of overfitting on small datasets if not done carefully (e.g., without freezing enough layers or using regularization).

## 3. Practical Implementation (PyTorch/TensorFlow/Keras)

Deep learning frameworks provide easy ways to implement transfer learning:

*   **Loading Pre-trained Models:** Libraries like `torchvision.models` (PyTorch) or `tf.keras.applications` (TensorFlow/Keras) offer popular architectures with weights pre-trained on ImageNet. (`pretrained=True` or `weights='imagenet'`).
*   **Modifying the Classifier Head:** Access the final layer (often named `fc`, `classifier`, or `predictions`) and replace it with a new layer matching your target number of classes.
*   **Freezing Layers:** Set the `requires_grad` attribute of layer parameters to `False` (PyTorch) or set `layer.trainable = False` (TensorFlow/Keras) to freeze them.
*   **Optimizer Configuration:** When fine-tuning different parts of the network with different learning rates (e.g., lower LR for convolutional base, higher LR for the new head), configure the optimizer accordingly.

## 4. Benefits of Transfer Learning

*   **Reduced Data Requirement:** Achieves good performance with much less task-specific labeled data.
*   **Faster Training:** Converges much faster as the network starts with meaningful weights instead of random initialization.
*   **Improved Performance:** Often leads to higher accuracy compared to training from scratch, especially with limited data.

## Next Steps

Transfer learning is widely used across many CV domains. Having covered core deep learning architectures and how to adapt them, we'll move on to applying these concepts to **Video Analysis**, looking at tasks like object tracking and action recognition. 