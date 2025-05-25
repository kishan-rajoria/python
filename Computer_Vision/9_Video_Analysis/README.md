# Topic 9: Video Analysis

## Overview

While previous topics focused primarily on static images, many real-world applications involve analyzing video streams. Video analysis introduces the temporal dimension – understanding how scenes and objects change over time. This topic covers key tasks in video analysis, including object tracking, action recognition, and optical flow estimation.

## 1. Representing Video Data

*   **Sequence of Frames:** A video is essentially a sequence of images (frames) displayed rapidly.
*   **Frame Rate:** The number of frames displayed per second (fps), determining the smoothness of motion.
*   **Temporal Information:** The key difference from static images is the relationship and motion between consecutive frames.

## 2. Object Tracking

Object tracking aims to locate a specific object (or multiple objects) across the frames of a video sequence. Given the initial location of an object in one frame, the goal is to follow its trajectory in subsequent frames.

*   **Challenges:** Appearance changes (illumination, pose, scale), occlusion, fast motion, interactions between multiple objects, camera motion.
*   **Approaches:**
    *   **Detection-Based Tracking ("Tracking-by-Detection"):**
        1.  Run an object detector (like YOLO, Faster R-CNN from Topic 5) independently on each frame.
        2.  Associate detections across frames based on criteria like spatial proximity (IoU overlap), appearance similarity (using features or embeddings), or motion prediction.
        *   *Pros:* Can handle objects entering/leaving the scene, robust to drift if the detector is reliable.
        *   *Cons:* Relies heavily on detector performance, association can be complex, computationally expensive if running detector on every frame.
    *   **Generative / Template-Based Tracking:**
        1.  Model the appearance of the target object in the first frame (template).
        2.  Search for the region in the next frame that best matches the template (e.g., using correlation filters or template matching).
        3.  Update the template based on the new location.
        *   *Pros:* Can be very fast, doesn't require detecting all objects first.
        *   *Cons:* Prone to drift if appearance changes significantly, struggles with occlusion, usually tracks only one object per tracker instance. Examples include MeanShift, CamShift.
    *   **Discriminative / Tracking-as-Classification:**
        1.  Train a classifier online to distinguish the target object from its surrounding background.
        2.  In the next frame, evaluate candidate regions using the classifier and select the one with the highest score.
        3.  Update the classifier based on the new location.
        *   *Pros:* Can adapt to appearance changes.
        *   *Cons:* Requires online training/updating, computationally more involved than template methods. Examples include MIL (Multiple Instance Learning), TLD (Tracking-Learning-Detection).
    *   **Deep Learning-Based Tracking:** Modern trackers often use deep learning features.
        *   *Siamese Networks:* Learn an embedding function to measure the similarity between a target template and candidate regions in subsequent frames.
        *   Integrating deep features into detection-based or discriminative frameworks.
*   **OpenCV Trackers:** OpenCV provides implementations of several classical tracking algorithms (`cv2.TrackerMIL_create`, `cv2.TrackerKCF_create`, `cv2.TrackerCSRT_create`, etc.).

## 3. Optical Flow

Optical flow estimates the apparent motion of objects or patterns between consecutive frames in a video. It calculates a vector for each pixel (or region) indicating its displacement (direction and magnitude) from one frame to the next.

*   **Assumptions:** Often relies on brightness constancy (a pixel's intensity doesn't change much between frames) and spatial smoothness (neighboring pixels move similarly).
*   **Types:**
    *   **Dense Optical Flow:** Calculates a flow vector for *every* pixel in the image. (e.g., Farneback method `cv2.calcOpticalFlowFarneback`).
    *   **Sparse Optical Flow:** Calculates flow vectors only for a sparse set of *interest points* (e.g., corners detected by Shi-Tomasi or Harris). (e.g., Lucas-Kanade method `cv2.calcOpticalFlowPyrLK`).
*   **Applications:** Motion estimation, video compression, object tracking support, camera motion analysis, activity recognition.

## 4. Action Recognition / Activity Recognition

This task involves classifying the action or activity being performed in a video clip (e.g., "walking", "running", "playing guitar", "handshake").

*   **Challenges:** Viewpoint variation, intra-class variation in how actions are performed, duration variability, background clutter, camera motion.
*   **Approaches:**
    *   **Frame-Based Methods:** Process individual frames using a 2D CNN (like ResNet) and aggregate the predictions (e.g., averaging, max pooling, using an RNN/LSTM over frame features). Simple but ignores temporal dynamics well.
    *   **Two-Stream Networks:** Use two separate CNNs:
        1.  **Spatial Stream:** Processes individual RGB frames to capture appearance information.
        2.  **Temporal Stream:** Processes stacked optical flow fields between frames to capture motion information.
        Fuse the predictions from both streams.
    *   **3D Convolutional Networks (C3D, I3D):** Apply convolutions in both spatial and temporal dimensions (`kernel_depth x height x width`). Allows the network to directly learn spatio-temporal features from raw video frames. Inflated 3D (I3D) networks inflate pre-trained 2D CNN filters (like ResNet) into 3D, enabling transfer learning from ImageNet.
    *   **Recurrent Networks on CNN Features:** Use LSTMs or GRUs on top of frame-level features extracted by a 2D CNN to model temporal dependencies.
    *   **Transformer-Based Methods (e.g., VideoMAE, TimeSformer):** Adapt the Transformer architecture for video by treating video as a sequence of frame patches or features, using spatio-temporal attention mechanisms.

## Next Steps

Understanding motion and actions in 2D video sequences is crucial. The next topic, **3D Vision & Scene Understanding**, explores techniques to infer the three-dimensional structure of the world from images or videos, enabling tasks like depth estimation and 3D reconstruction. 