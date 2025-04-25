# Topic 4: Image Segmentation

## Overview

Image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, also known as superpixels or image objects). The goal of segmentation is typically to simplify or change the representation of an image into something that is more meaningful and easier to analyze. Each pixel in an image is assigned a label such that pixels with the same label share certain visual characteristics (e.g., color, intensity, texture).

Segmentation is a crucial step in many CV applications, including medical imaging (locating tumors or tissues), autonomous driving (identifying roads, pedestrians, vehicles), satellite imagery analysis, and more.

## 1. Thresholding-Based Segmentation

One of the simplest methods, often applied to grayscale images.

*   **Global Thresholding:** Select a single threshold value `T`. All pixels with intensity greater than `T` belong to one segment (e.g., foreground), and the rest belong to another (e.g., background).
    *   Choosing `T` can be done manually or automatically.
    *   **Otsu's Binarization:** An automatic method that finds the optimal threshold value by minimizing the intra-class variance (or maximizing the inter-class variance) between the two groups of pixels separated by the threshold. Assumes a bimodal histogram. (See `cv2.threshold` with `cv2.THRESH_OTSU`).
*   **Adaptive Thresholding:** Computes the threshold for small regions of the image, rather than globally. This is useful when illumination varies across the image.
    *   *Methods:* Mean thresholding (threshold is the mean of the neighborhood area) or Gaussian thresholding (threshold is a weighted sum of neighborhood values, weights are Gaussian).
    *   (See `cv2.adaptiveThreshold`).

## 2. Region-Based Segmentation

These methods group pixels into regions based on similarity criteria.

*   **Region Growing:**
    *   *Concept:* Starts with initial "seed" pixels. Iteratively adds neighboring pixels to a region if they satisfy a similarity criterion (e.g., intensity difference within a tolerance, similar color).
    *   *Process:* Requires selecting seed points and defining a suitable similarity measure. Growth stops when no more pixels can be added to any region.
*   **Region Splitting and Merging:**
    *   *Concept:* A top-down (splitting) and bottom-up (merging) approach.
    *   *Process:* Start with the whole image as one region. If a region is not homogeneous (based on some criterion), split it (e.g., into quadrants). Repeat recursively. Then, merge adjacent regions if they are similar enough.

## 3. Clustering-Based Segmentation

Treats segmentation as a clustering problem, where pixels with similar features (color, intensity, texture, location) are grouped together.

*   **K-Means Clustering:**
    *   *Concept:* Partitions pixels into `K` clusters, where `K` is predefined. Each pixel belongs to the cluster with the nearest mean (cluster center).
    *   *Mechanism:* Iteratively assigns pixels to the nearest cluster centroid and recalculates the centroids based on the assigned pixels. Often applied to pixel color values (e.g., in RGB or LAB color space).
    *   (See `cv2.kmeans`).

## 4. Edge-Based Segmentation

Uses edge information (detected using methods from Topic 2, like Canny) to identify boundaries between segments.

*   *Concept:* Assumes that segment boundaries correspond to strong edges in the image.
*   *Challenge:* Detected edges are often disconnected or noisy, requiring linking and post-processing steps to form closed boundaries and define segments. Watershed algorithms are sometimes used in conjunction with edge maps.

## 5. Watershed Algorithm

*   *Concept:* Visualizes the image as a topographic landscape where intensity represents height. "Flooding" this landscape from local minima creates catchment basins. The lines where different basins meet ("watershed lines") represent segment boundaries.
*   *Application:* Often used with gradient images (where edges have high intensity) or distance transforms (for separating touching objects). Requires careful marker selection (initial minima/sources) to avoid over-segmentation.
*   (See `cv2.watershed`).

## 6. Introduction to Deep Learning-Based Segmentation

Modern state-of-the-art segmentation relies heavily on deep learning, particularly Convolutional Neural Networks (CNNs).

*   **Semantic Segmentation:** Assigns a class label (e.g., "car", "road", "sky", "person") to *each pixel* in the image.
    *   *Architectures:* Often use Fully Convolutional Networks (FCNs), U-Net, DeepLab. These typically involve an encoder (downsampling path to capture context) and a decoder (upsampling path to produce a full-resolution segmentation map).
*   **Instance Segmentation:** Goes a step further than semantic segmentation. It not only classifies each pixel but also distinguishes between different *instances* of the same object class (e.g., labeling "car 1", "car 2", "car 3" distinctly).
    *   *Architectures:* Often build upon object detectors, like Mask R-CNN.

*We will cover these deep learning methods in more detail in later topics.*

## Next Steps

Understanding how to partition images sets the stage for **Object Detection**, where the goal is not just to segment regions but to identify specific object instances within an image and draw bounding boxes around them. 