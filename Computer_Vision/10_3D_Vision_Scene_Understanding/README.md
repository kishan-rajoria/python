# Topic 10: 3D Vision & Scene Understanding

## Overview

While 2D computer vision focuses on analyzing flat images, many applications require understanding the three-dimensional structure of the world. 3D Vision deals with capturing, processing, and interpreting 3D information from visual data. Scene understanding aims to build a holistic interpretation of a scene, including object recognition, localization (in 3D), relationships between objects, and geometric layout.

## 1. Camera Geometry & Calibration

Understanding how a 3D world point projects onto a 2D image sensor is fundamental.

*   **Pinhole Camera Model:** A simplified mathematical model describing the relationship between 3D world coordinates and 2D image coordinates.
*   **Intrinsic Parameters:** Describe the internal properties of the camera (focal length `fx, fy`, optical center/principal point `cx, cy`, skew). Stored in the camera matrix `K`.
*   **Extrinsic Parameters:** Describe the camera's position and orientation in the 3D world (rotation `R` and translation `t`).
*   **Camera Calibration:** The process of estimating the intrinsic and extrinsic parameters of a camera, typically using images of a known calibration pattern (e.g., a chessboard). (See `cv2.calibrateCamera`). Essential for accurate 3D measurements.

## 2. Stereo Vision

Uses two (or more) cameras viewing the same scene from slightly different viewpoints to infer depth, similar to human binocular vision.

*   **Epipolar Geometry:** The geometric relationship between two views of a scene. Describes the constraints on where a point in one image can appear in the other image (epipolar lines). Key concepts include the **epipole** and the **fundamental matrix** `F` (or **essential matrix** `E` if intrinsics are known).
*   **Stereo Correspondence (Matching):** The core challenge is finding corresponding pixels (representing the same 3D point) in the left and right images. This is often done by comparing image patches along epipolar lines using similarity measures (e.g., Sum of Squared Differences - SSD, Normalized Cross-Correlation - NCC).
*   **Disparity:** The difference in horizontal coordinates between corresponding points in the left and right images (after stereo rectification). Disparity is inversely proportional to depth (`Depth ∝ 1 / Disparity`).
*   **Stereo Rectification:** A process that warps the images such that epipolar lines become horizontal scanlines, simplifying the correspondence search to a 1D problem.
*   **Depth Map:** An image where pixel values represent the estimated depth (or disparity) of the corresponding point in the scene.
*   **Algorithms:** Block Matching, Semi-Global Block Matching (SGBM). (See `cv2.StereoBM_create`, `cv2.StereoSGBM_create`).

## 3. Structure from Motion (SfM)

Reconstructs the 3D structure of a scene *and* the camera poses from a sequence of 2D images taken from *unknown* viewpoints (unlike stereo where camera poses are usually known/calibrated).

*   **Process Overview:**
    1.  **Feature Detection & Matching:** Detect and match keypoints (e.g., SIFT, ORB) across multiple images (Topic 3).
    2.  **Geometric Verification:** Use the fundamental matrix (estimated from matches, e.g., using RANSAC) to filter out incorrect matches.
    3.  **Initialization:** Estimate the initial camera poses and 3D points for a small subset of views (often a stereo pair).
    4.  **Triangulation:** Given known camera poses and corresponding 2D points, estimate the 3D location of the points.
    5.  **Bundle Adjustment:** A large non-linear optimization problem that jointly refines the 3D point locations *and* all camera parameters (poses and intrinsics) to minimize the reprojection error (the difference between observed 2D feature locations and the projection of the estimated 3D points). This is the most crucial and computationally intensive step.
    6.  **Incremental Reconstruction:** Add more views one by one, estimating their pose (PnP - Perspective-n-Point problem), triangulating new points, and running bundle adjustment periodically.
*   **Libraries:** COLMAP, OpenMVG, AliceVision/Meshroom. OpenCV provides building blocks but not a full SfM pipeline.

## 4. SLAM (Simultaneous Localization and Mapping)

Primarily used in robotics and AR/VR, SLAM aims to build a map of an unknown environment while simultaneously tracking the agent's (e.g., robot, camera) location within that map, often in real-time.

*   **Key Difference from Offline SfM:** SLAM operates online and sequentially, using sensor data (often cameras, sometimes IMUs, LiDAR) as it arrives. Efficiency is critical.
*   **Components:**
    *   **Tracking:** Estimates the camera pose for the current frame relative to the existing map.
    *   **Mapping:** Updates the map (e.g., 3D point cloud, landmarks) using information from the current frame.
    *   **Loop Closure Detection:** Recognizes previously visited locations to correct accumulated drift in pose estimation and map consistency.
    *   **(Optional) Relocalization:** Recovers the camera pose if tracking is lost.
*   **Types:**
    *   **Visual SLAM (vSLAM):** Uses cameras as the primary sensor. Feature-based (using keypoints like ORB) and direct methods (using pixel intensities directly) exist. Examples: ORB-SLAM, DSO.
    *   **Visual-Inertial SLAM (VI-SLAM):** Fuses visual data with measurements from an Inertial Measurement Unit (IMU) for improved robustness, especially during fast motion or in texture-less environments. Examples: VINS-Mono, OKVIS.

## 5. 3D Data Representations

*   **Point Clouds:** Collections of 3D points (X, Y, Z coordinates), possibly with additional attributes like color or normals. Simple but unstructured.
*   **Meshes:** Represent surfaces using vertices, edges, and faces (typically triangles or quadrilaterals). Provide connectivity information.
*   **Voxel Grids:** Divide 3D space into a regular grid of volumetric pixels (voxels). Can represent occupancy or other properties.
*   **Implicit Representations (e.g., NeRF - Neural Radiance Fields):** Use neural networks to learn a continuous function that maps 3D coordinates (and viewing directions) to color and density, allowing for novel view synthesis.

## 6. Scene Understanding

Goes beyond geometry to interpret the semantic content and relationships within a scene. Combines techniques like:

*   **3D Object Detection:** Locating objects and estimating their 3D bounding boxes.
*   **3D Semantic Segmentation:** Assigning a semantic label to each point or voxel in a 3D representation.
*   **Layout Estimation:** Inferring the overall structure of the scene (e.g., room layout, ground plane).
*   **Relationship Reasoning:** Understanding how objects interact or relate to each other spatially and semantically.

## Next Steps

Having explored methods for interpreting 2D images, videos, and 3D structures, the final topic covers **Advanced Topics & Future Trends**, including generative models for vision, self-supervised learning, the intersection of vision and language, and ethical considerations in computer vision. 