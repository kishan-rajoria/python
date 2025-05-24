# Topic 7: Advanced CNN Architectures

## Overview

While foundational architectures like LeNet, AlexNet, and VGGNet established the power of CNNs, research quickly focused on building deeper and more efficient networks to tackle increasingly complex visual tasks and larger datasets like ImageNet. This topic explores key architectural innovations that enabled these advancements.

## 1. Challenges of Deep Networks

Simply stacking more layers (like in VGG) leads to problems:

*   **Vanishing/Exploding Gradients:** As gradients are backpropagated through many layers, they can become extremely small (vanish) or large (explode), hindering learning in early layers.
*   **Degradation Problem:** Counter-intuitively, very deep "plain" networks (simple stacks of layers) often exhibit *higher* training error than their shallower counterparts. This isn't just overfitting; the deeper models struggle to learn even the identity function for the extra layers.

## 2. Residual Networks (ResNets)

ResNet (He et al., 2015) was a landmark architecture that enabled training significantly deeper networks (hundreds or even over a thousand layers) by addressing the degradation problem.

*   **Key Idea: Residual Learning:** Instead of trying to learn an underlying mapping `H(x)` directly with a stack of layers, learn the *residual* mapping `F(x) = H(x) - x`. The original mapping is then reconstructed as `H(x) = F(x) + x`.
*   **Residual Block (Identity Shortcut):** Implemented using "shortcut connections" or "skip connections". The input `x` to a block of layers (e.g., two Conv layers) is added directly to the output `F(x)` of that block (element-wise addition).
    ![Residual Block](https://miro.medium.com/max/1400/1*q4U44H_DSL3EhH76nN-G4A.png)
    *(Image Source: Kaiming He et al., "Deep Residual Learning for Image Recognition")*
*   **Benefit:** If the optimal function for the added layers is close to the identity mapping (`H(x) = x`), it's much easier for the network to learn to drive the residual `F(x)` towards zero than to learn the identity mapping directly with a stack of non-linear layers. This allows gradients to flow more easily through the network and makes it feasible to train much deeper models without degradation.
*   **Impact:** ResNets became the backbone for countless subsequent CV models due to their effectiveness and ability to scale depth. Many variants exist (ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152, ResNeXt, Wide ResNets).

## 3. Inception Networks (GoogLeNet)

The Inception architecture (Szegedy et al., 2014), also known as GoogLeNet (winner of ILSVRC 2014), focused on improving computational efficiency and performance by using multi-scale processing within network layers.

*   **Key Idea: Inception Module:** Instead of choosing just one filter size (e.g., 3x3) or pooling for a layer, apply multiple filter sizes (1x1, 3x3, 5x5) and max pooling *in parallel* to the same input feature map. Concatenate the outputs of these parallel branches along the channel dimension.
    ![Inception Module](https://miro.medium.com/max/1400/1*GCzSzMEjh-XANU8MVQxcCQ.jpeg)
    *(Image Source: Christian Szegedy et al., "Going Deeper with Convolutions")*
*   **1x1 Convolutions (Bottleneck Layers):** Used extensively within Inception modules *before* the larger 3x3 and 5x5 convolutions, and *after* max pooling. These act as dimensionality reduction layers, significantly reducing the computational cost without sacrificing much performance. They reduce the number of input channels (depth) fed into the expensive larger convolutions.
*   **Benefit:** Allows the network to capture features at multiple scales simultaneously within the same layer. Computationally more efficient than simply using very large filters. Led to state-of-the-art results with fewer parameters than VGG.
*   **Evolution:** Several versions exist (Inception v1/GoogLeNet, Inception v2/v3, Inception v4, Inception-ResNet).

## 4. Efficient Architectures (MobileNets, ShuffleNets, etc.)

As models became deeper, deploying them on resource-constrained devices (like mobile phones) became a challenge. This led to architectures focusing on efficiency (reducing parameters and computational cost - FLOPs).

*   **MobileNets (Howard et al., 2017):**
    *   **Key Idea: Depthwise Separable Convolutions:** Replaces standard convolution with two steps:
        1.  **Depthwise Convolution:** Applies a *single* filter to each input channel independently.
        2.  **Pointwise Convolution (1x1 Convolution):** Combines the outputs of the depthwise convolution across channels.
    *   **Benefit:** Dramatically reduces computation and model size compared to standard convolutions with similar output depth. MobileNetV1, V2, and V3 introduced further improvements (e.g., inverted residuals, linear bottlenecks, squeeze-and-excitation, neural architecture search).

*   **ShuffleNets (Zhang et al., 2017):**
    *   **Key Idea:** Uses pointwise group convolutions and channel shuffling to further reduce computational cost while maintaining information flow between channel groups.

*   **SqueezeNets, EfficientNets, etc.:** Many other architectures explore different ways to achieve high accuracy with significantly fewer parameters and FLOPs, often using techniques like neural architecture search (NAS) to find optimal structures.

## 5. Vision Transformers (ViT)

Inspired by the success of Transformers in NLP, Vision Transformers (Dosovitskiy et al., 2020) adapt the Transformer architecture for image classification.

*   **Key Idea:** Treat an image as a sequence of patches.
    1.  Split the image into fixed-size patches (e.g., 16x16 pixels).
    2.  Linearly embed each patch into a vector.
    3.  Add positional embeddings to retain spatial information (similar to NLP Transformers).
    4.  Feed this sequence of patch embeddings into a standard Transformer encoder (multi-head self-attention, feed-forward layers).
    5.  Use a classification head (usually attached to an extra learnable "class token" prepended to the sequence) for the final prediction.
*   **Performance:** When pre-trained on very large datasets (like JFT-300M or ImageNet-21k), ViTs can achieve state-of-the-art results, often outperforming CNNs. They tend to require more data than CNNs to generalize well if trained from scratch on smaller datasets like ImageNet-1k.
*   **Impact:** Sparked significant research into applying Transformer-based models to various vision tasks beyond classification. Hybrid CNN-Transformer models also exist.

## Next Steps

These advanced architectures often achieve their best performance when leveraging knowledge learned from massive datasets. The next topic, **Transfer Learning & Fine-tuning for CV**, explores how to adapt these powerful pre-trained models for specific computer vision tasks, even with limited task-specific data. 