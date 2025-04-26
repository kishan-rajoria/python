# Topic 5: Object Detection

## Overview

Object detection is a fundamental computer vision task that deals with identifying the presence, location, and type of one or more objects within an image or video. Unlike image classification (which assigns a single label to the whole image), object detection provides:

1.  **Classification:** What objects are present (e.g., "car", "person", "dog").
2.  **Localization:** Where these objects are located, typically represented by bounding boxes (rectangles defined by coordinates like top-left corner, width, and height).

Object detection is critical for applications like autonomous driving, surveillance, robotics, image retrieval, and human-computer interaction.

## 1. Traditional Approaches (Pre-Deep Learning)

Early methods often involved multi-stage processes:

*   **Sliding Window:**
    *   *Concept:* Define windows (bounding boxes) of various sizes and aspect ratios. Slide these windows across the entire image at different scales.
    *   *Classification:* For each window, extract features (e.g., HOG - Histogram of Oriented Gradients, Haar features) and use a classifier (e.g., Support Vector Machine - SVM, AdaBoost) to determine if an object of interest is present within that window.
    *   *Challenges:* Computationally very expensive due to the vast number of windows to check. Difficulty in precisely localizing objects and handling varying aspect ratios.

*   **Viola-Jones Detector (Haar Cascades):**
    *   *Concept (Viola & Jones, 2001):* A highly influential and efficient algorithm, primarily known for real-time face detection.
    *   *Key Ideas:*
        *   **Haar-like Features:** Simple rectangular features (differences of sums of pixel intensities in adjacent regions) that are fast to compute using *integral images*.
        *   **Integral Image:** A representation allowing rapid calculation of sums of pixel values within any rectangular region.
        *   **AdaBoost (Adaptive Boosting):** A machine learning algorithm used to select a small number of critical Haar features from a vast pool and combine them into a strong classifier.
        *   **Cascade Classifier:** Arrange classifiers in stages (a cascade). Simpler stages with fewer features quickly reject non-object windows, while subsequent stages become progressively more complex to evaluate promising regions. This significantly speeds up detection by avoiding costly computations on most background windows.
    *   *Pros:* Very fast for specific object classes it's trained for (like faces).
    *   *Cons:* Less generalizable than deep learning methods, sensitive to orientation, requires careful feature engineering and training for new object classes.
    *   (See `cv2.CascadeClassifier`).

*   **Histogram of Oriented Gradients (HOG) with SVM:**
    *   *Concept:* HOG features capture local gradient orientation distributions. They proved effective for pedestrian detection when combined with a linear SVM classifier within a sliding window framework.
    *   *Pros:* More robust to illumination changes than Haar features.
    *   *Cons:* Still relies on sliding windows, less effective than deep learning.

## 2. Deep Learning-Based Object Detectors

Modern object detection is dominated by deep learning, specifically Convolutional Neural Networks (CNNs). These methods typically integrate feature extraction, region proposal (or direct prediction), classification, and bounding box regression into end-to-end trainable networks.

### a) Two-Stage Detectors (Region Proposal Networks)

These methods first propose candidate regions likely to contain objects and then classify/refine these regions.

*   **R-CNN (Regions with CNN features):**
    *   *Concept:* Uses Selective Search (a traditional method) to generate ~2000 region proposals. Warps each region to a fixed size and feeds it into a CNN for feature extraction. Classifies features using SVMs and refines bounding boxes using linear regression.
    *   *Cons:* Very slow due to processing each region independently. Training is multi-stage and complex.
*   **Fast R-CNN:**
    *   *Concept:* Feeds the *entire image* through a CNN once to get a feature map. Projects region proposals onto this feature map. Uses RoIPooling (Region of Interest Pooling) to extract fixed-size feature vectors for each proposal. Feeds these into fully connected layers for classification and bounding box regression.
    *   *Pros:* Much faster than R-CNN. End-to-end training (mostly).
*   **Faster R-CNN:**
    *   *Concept:* Introduces a **Region Proposal Network (RPN)**, a small neural network that *learns* to generate region proposals directly from the CNN feature map, eliminating the bottleneck of external methods like Selective Search. The RPN predicts objectness scores and bounding box anchors.
    *   *Pros:* Significantly faster than Fast R-CNN, state-of-the-art performance for its time. Fully end-to-end trainable. Basis for many subsequent detectors.
*   **Mask R-CNN:**
    *   *Concept:* Extends Faster R-CNN to perform **instance segmentation** by adding a parallel branch that predicts a binary mask (segmenting the object within the bounding box) for each detected object. Uses RoIAlign instead of RoIPooling for more precise feature extraction needed for masks.

### b) One-Stage Detectors (Single Shot Detectors)

These methods predict bounding boxes and class probabilities directly from the full feature map in a single pass, without a separate region proposal step.

*   **YOLO (You Only Look Once):**
    *   *Concept:* Divides the image into a grid. Each grid cell is responsible for predicting bounding boxes and class probabilities for objects whose center falls within that cell. Predicts multiple boxes and confidence scores per cell.
    *   *Pros:* Extremely fast (real-time detection). Sees the entire image context when making predictions.
    *   *Cons:* Early versions struggled with small objects and precise localization compared to two-stage detectors. Many versions exist (YOLOv2, YOLOv3, YOLOv4, YOLOR, YOLOX, YOLOv5, YOLOv7, YOLOv8...).
*   **SSD (Single Shot MultiBox Detector):**
    *   *Concept:* Predicts bounding boxes and class scores using small convolutional filters applied to feature maps from multiple layers of a base network. Uses default boxes (anchors) of different aspect ratios and scales at different feature map resolutions to handle objects of varying sizes.
    *   *Pros:* Fast, good balance between speed and accuracy.
*   **RetinaNet:**
    *   *Concept:* Addressed the class imbalance issue (vast number of easy negative background boxes vs. few positive object boxes) in one-stage detectors using **Focal Loss**. Focal Loss down-weights the loss contribution from easy negatives, allowing the model to focus on learning hard examples. Combined with a Feature Pyramid Network (FPN) for multi-scale feature extraction.
    *   *Pros:* Achieved accuracy comparable to two-stage detectors while maintaining one-stage speed.

## 3. Common Concepts in Deep Learning Detectors

*   **Backbone Network:** A pre-trained CNN (e.g., VGG, ResNet, MobileNet) used for initial feature extraction from the input image.
*   **Neck:** Intermediate layers connecting the backbone and the head (e.g., Feature Pyramid Network - FPN) used to aggregate features from multiple scales.
*   **Head:** The final layers responsible for predicting class probabilities and bounding box coordinates.
*   **Anchor Boxes (Default Boxes):** Predefined boxes of various sizes and aspect ratios used as references. The network predicts offsets relative to these anchors and classifies whether an object is present in each anchor.
*   **Non-Maximum Suppression (NMS):** A post-processing step to eliminate highly overlapping bounding boxes predicted for the same object, keeping only the one with the highest confidence score.
*   **Intersection over Union (IoU):** A metric used to measure the overlap between a predicted bounding box and the ground truth bounding box. Crucial for evaluating detector performance and for training (assigning anchors to ground truth).

## Next Steps

Object detection often relies heavily on strong feature extraction capabilities. The next topic, **Image Classification & Deep Learning Basics**, will delve deeper into the architecture and training of Convolutional Neural Networks (CNNs), which form the backbone of most modern object detectors and other CV tasks. 