# Ensure required libraries are installed:
# pip install opencv-python numpy

import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")

print("\n--- 3D Triangulation Demonstration --- ")
print("Estimating the 3D position of a point from two 2D views with known camera poses.")

# --- 1. Define Camera Intrinsic Parameters (K Matrices) --- 
# These are typically obtained through camera calibration.
# We'll use plausible example values.
focal_length = 600 # Example focal length in pixels
cx = 320 # Example principal point x (image center)
cy = 240 # Example principal point y (image center)

K1 = np.array([[focal_length, 0, cx],
               [0, focal_length, cy],
               [0, 0, 1]], dtype=np.float32)

K2 = K1.copy() # Assuming identical cameras for simplicity

print("\nCamera Intrinsic Matrix (K):")
print(K1)

# --- 2. Define Camera Poses (Extrinsic Parameters) --- 
# Pose of Camera 1: Assumed to be at the origin (identity matrix)
R1 = np.identity(3, dtype=np.float32)
t1 = np.zeros((3, 1), dtype=np.float32)

# Pose of Camera 2: Relative to Camera 1 (e.g., shifted along x-axis for stereo)
baseline = 0.1 # Example baseline distance (e.g., 10cm = 0.1m)
R2 = np.identity(3, dtype=np.float32) # Assuming cameras are parallel
t2 = np.array([[baseline], [0], [0]], dtype=np.float32) # Translation along X

print("\nCamera 1 Pose (R1, t1): World Origin")
print("Camera 2 Pose (R2, t2): Relative to Camera 1")
print(f"  R2:\n{R2}")
print(f"  t2:\n{t2}")

# --- 3. Define Sample 2D Projections --- 
# These are the coordinates (u, v) of the SAME 3D point as seen in each camera's image plane.
# In a real scenario, these points would come from feature matching.

# Example: A 3D point at (X=0.2, Y=0.1, Z=1.5) meters relative to Camera 1
# We can project this point to get realistic 2D points (optional check)
# P_world = np.array([[0.2], [0.1], [1.5]])
# P_cam1_hom = K1 @ (R1 @ P_world + t1) # Project into Cam1
# p1_hom = P_cam1_hom / P_cam1_hom[2]   # Normalize
# P_cam2_hom = K2 @ (R2 @ P_world + t2) # Project into Cam2
# p2_hom = P_cam2_hom / P_cam2_hom[2]   # Normalize
# print(f"\nProjected 2D point in Cam1 (ideal): {p1_hom[:2].flatten()}")
# print(f"Projected 2D point in Cam2 (ideal): {p2_hom[:2].flatten()}")

# Let's use slightly noisy versions of these ideal points:
point1_2D = np.array([[400.5], [280.2]], dtype=np.float32) # (u1, v1) in Camera 1
point2_2D = np.array([[360.1], [279.8]], dtype=np.float32) # (u2, v2) in Camera 2

print(f"\nObserved 2D point projection in Camera 1: {point1_2D.flatten()}")
print(f"Observed 2D point projection in Camera 2: {point2_2D.flatten()}")

# --- 4. Construct Projection Matrices (P = K * [R|t]) --- 
# OpenCV format: P is 3x4
# [R|t] is the transformation from World to Camera coordinates (3x4)
Rt1 = np.hstack((R1, t1))
P1 = K1 @ Rt1 # Projection matrix for Camera 1

Rt2 = np.hstack((R2, t2))
P2 = K2 @ Rt2 # Projection matrix for Camera 2

print("\nProjection Matrix P1 (K1 * [R1|t1]):")
print(P1)
print("\nProjection Matrix P2 (K2 * [R2|t2]):")
print(P2)

# --- 5. Triangulate 3D Point --- 
# Input points for triangulatePoints should be 2xN or Nx2 (N=number of points)
# We have N=1 point here.
# Reshape points to 2x1
points1 = point1_2D.reshape(2, 1)
points2 = point2_2D.reshape(2, 1)

# Perform triangulation
# Output is 4xN array of homogeneous coordinates
point4D_hom = cv2.triangulatePoints(P1, P2, points1, points2)

print("\nTriangulation Result (Homogeneous 4D coordinates):")
print(point4D_hom)

# --- 6. Convert to Non-homogeneous Coordinates --- 
# Divide by the 4th coordinate (w) to get 3D coordinates (X, Y, Z)
point3D = point4D_hom / point4D_hom[3]

print("\nEstimated 3D Point (X, Y, Z) relative to Camera 1 frame:")
print(point3D[:3].flatten()) # Print X, Y, Z

print("\nNote: The accuracy depends heavily on the calibration accuracy (K, R, t)")
print("and the precision of the 2D point correspondences.")

print("\nScript finished.") 