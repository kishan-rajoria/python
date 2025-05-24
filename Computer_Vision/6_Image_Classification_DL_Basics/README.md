# Topic 6: Image Classification & Deep Learning Basics

## Overview

Image classification is the task of assigning a single label (representing a class) to an entire image (e.g., "cat", "dog", "car"). While traditional methods used hand-crafted features (like SIFT or HOG) fed into classifiers (like SVMs), the advent of Deep Learning, particularly Convolutional Neural Networks (CNNs), revolutionized the field, achieving state-of-the-art performance. This topic introduces the fundamental concepts of CNNs and the basics of training deep learning models for image classification.

## 1. Limitations of Traditional ML for Images

Applying traditional machine learning (like fully connected neural networks or SVMs) directly to raw pixel values faces significant challenges:

*   **High Dimensionality:** Images contain a vast number of pixels (e.g., a 224x224 RGB image has 224*224*3 = 150,528 features). This leads to computationally expensive models and the "curse of dimensionality".
*   **Spatial Structure Ignorance:** Fully connected networks treat pixels independently, ignoring the crucial spatial relationships between neighboring pixels (e.g., pixels forming an edge or texture).
*   **Lack of Invariance:** They are not inherently invariant to translation (shifting the object within the image), scale, or viewpoint changes. Slight shifts can dramatically change the input feature vector.

## 2. Convolutional Neural Networks (CNNs / ConvNets)

CNNs are a specialized type of neural network designed to process grid-like data, such as images, by leveraging principles like parameter sharing and spatial hierarchies.

### a) Core Building Blocks

*   **Convolutional Layer:**
    *   *Concept:* Applies learnable filters (kernels) across the input image (or feature map from a previous layer) using the convolution operation (Topic 2).
    *   *Function:* Detects local patterns (edges, corners, textures in early layers; more complex motifs in deeper layers).
    *   *Key Ideas:*
        *   **Local Receptive Fields:** Each neuron in the output feature map connects only to a small region (the receptive field) of the input.
        *   **Parameter Sharing:** The *same* filter (set of weights) is applied across the entire input, significantly reducing the number of parameters compared to fully connected layers and allowing the detection of a pattern regardless of its position (translation equivariance).
        *   **Feature Maps:** The output of applying one filter across the input. A convolutional layer typically learns multiple filters, producing multiple feature maps, each detecting different patterns.
    *   *Hyperparameters:* Filter size (e.g., 3x3, 5x5), number of filters, stride (step size of the filter), padding (handling borders).

*   **Activation Function (e.g., ReLU):**
    *   *Concept:* Introduces non-linearity into the model, allowing it to learn complex relationships. Applied element-wise after the convolution.
    *   *ReLU (Rectified Linear Unit):* `f(x) = max(0, x)`. Computationally efficient and helps mitigate vanishing gradients. Most commonly used activation in CNNs.

*   **Pooling Layer (Subsampling):**
    *   *Concept:* Reduces the spatial dimensions (width, height) of the feature maps, making the representation more robust to small translations and reducing computational cost.
    *   *Common Types:*
        *   **Max Pooling:** Takes the maximum value within a small window (e.g., 2x2). Retains the strongest activation for a feature.
        *   **Average Pooling:** Takes the average value within the window.
    *   *Function:* Creates a degree of local translation invariance and reduces the number of parameters for subsequent layers.
    *   *Hyperparameters:* Pooling window size, stride.

*   **Fully Connected Layer (Dense Layer):**
    *   *Concept:* Standard neural network layer where each neuron is connected to *all* neurons in the previous layer.
    *   *Function:* Typically used at the end of the CNN architecture after several convolutional and pooling layers. Takes the high-level features extracted by the convolutional part and uses them for the final classification task. The output layer usually has neurons equal to the number of classes, often followed by a Softmax activation to produce probability scores.

### b) Typical CNN Architecture

A common pattern for image classification CNNs:
`INPUT -> [[CONV -> ReLU] * N -> POOL?] * M -> [FC -> ReLU] * K -> FC (Softmax)`
Where `N`, `M`, `K` represent repetitions of blocks, and `POOL?` indicates pooling might be optional in some blocks. Early layers detect simple features, and deeper layers combine these to detect more complex patterns and objects.

## 3. Training CNNs

Training involves optimizing the network's weights (filter values, fully connected layer weights) to minimize the difference between its predictions and the true labels on a training dataset.

*   **Loss Function:** Measures the discrepancy between the predicted output and the ground truth label.
    *   *Common Choice for Classification:* **Cross-Entropy Loss** (measures the difference between two probability distributions).
*   **Backpropagation:** An algorithm to compute the gradients (derivatives) of the loss function with respect to each weight in the network.
*   **Optimizer:** An algorithm that updates the network's weights based on the computed gradients to minimize the loss function.
    *   *Common Choices:*
        *   **Stochastic Gradient Descent (SGD):** Updates weights based on the gradient computed from a small batch of data. Often used with momentum.
        *   **Adam (Adaptive Moment Estimation):** An adaptive learning rate optimization algorithm that is often faster to converge than SGD.
*   **Key Concepts:**
    *   **Epoch:** One complete pass through the entire training dataset.
    *   **Batch Size:** The number of training examples used in one iteration (forward/backward pass) to compute the gradient.
    *   **Learning Rate:** Controls the step size taken during weight updates. A crucial hyperparameter.
    *   **Overfitting:** When the model learns the training data too well, including its noise, and performs poorly on unseen data. Regularization techniques (like dropout, weight decay) and data augmentation are used to combat this.
    *   **Data Augmentation:** Artificially increasing the size and diversity of the training dataset by applying random transformations (rotation, flipping, cropping, color jittering) to the images. Helps the model generalize better.

## 4. Pioneering CNN Architectures

*   **LeNet-5 (LeCun et al., 1998):** One of the earliest successful CNNs, primarily used for handwritten digit recognition (MNIST). Established the basic structure of CONV -> POOL -> CONV -> POOL -> FC -> FC -> OUTPUT.
*   **AlexNet (Krizhevsky et al., 2012):** Won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 by a large margin, marking the resurgence of CNNs. Deeper than LeNet, used ReLU activation, dropout for regularization, and GPU acceleration.
*   **VGGNet (Simonyan & Zisserman, 2014):** Showed that depth is critical. Used a very simple and uniform architecture with stacks of small (3x3) convolutional filters followed by max pooling. Very deep (VGG-16, VGG-19) but has many parameters.

## Next Steps

Building upon these foundational CNN concepts, the next topic explores **Advanced CNN Architectures** like ResNet and Inception, which introduced techniques to train even deeper and more effective networks, along with architectures optimized for efficiency like MobileNets and the recent Vision Transformers (ViTs). 