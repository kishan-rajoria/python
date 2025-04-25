# Ensure required libraries are installed:
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
from matplotlib import pyplot as plt

print(f"OpenCV version: {cv2.__version__}")

# --- 1. Load/Create a Sample Image ---
# For feature detection, an image with texture/corners is better than a plain one.
# Let's create a simple image with a square and some lines.
img_size = 200
img = np.ones((img_size, img_size), dtype=np.uint8) * 200 # Gray background
# Draw a white square
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
# Draw some black lines
cv2.line(img, (20, 180), (180, 20), 0, 3)
cv2.line(img, (20, 20), (180, 180), 50, 2)

img_color_for_drawing = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Color version for drawing keypoints

print("\n--- Sample Image Created ---")
print(f"Image shape: {img.shape}")

# Optional: Display original image
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')
# plt.show()

# --- 2. Corner Detection ---
print("\n--- Corner Detection --- ")

# a) Harris Corner Detection
# Input should be float32 grayscale
gray_float = np.float32(img)
blockSize = 2 # Size of neighborhood considered
apertureSize = 3 # Aperture parameter for Sobel operator
k = 0.04 # Harris detector free parameter

harris_corners = cv2.cornerHarris(gray_float, blockSize, apertureSize, k)

# Result is dilated for marking the corners, not important for understanding
harris_corners_dilated = cv2.dilate(harris_corners, None)

# Threshold for an optimal value, it may vary depending on the image.
harris_threshold = 0.01 * harris_corners.max()
img_harris = img_color_for_drawing.copy()
img_harris[harris_corners_dilated > harris_threshold] = [0, 0, 255] # Mark corners in red

print(f"Harris: Detected potential corners (thresholded response > {harris_threshold:.4f})")
# cv2.imshow('Harris Corners', img_harris)

# b) Shi-Tomasi Corner Detection (Good Features to Track)
maxCorners = 100 # Max number of corners to return
qualityLevel = 0.01 # Minimal accepted quality of image corners
minDistance = 10 # Minimum possible Euclidean distance between the returned corners

corners = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance)
corners = np.intp(corners) # Use np.intp for recent numpy versions

img_shi_tomasi = img_color_for_drawing.copy()
print(f"Shi-Tomasi: Found {len(corners)} corners.")
for i in corners:
    x, y = i.ravel()
    cv2.circle(img_shi_tomasi, (x, y), 3, (0, 255, 0), -1) # Mark corners in green

# cv2.imshow('Shi-Tomasi Corners', img_shi_tomasi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --- 3. ORB (Oriented FAST and Rotated BRIEF) ---
print("\n--- ORB Keypoint Detection & Description --- ")

# Initialize ORB detector
# You can specify number of features, scale factor, pyramid levels etc.
orb = cv2.ORB_create(nfeatures=500) # Limit features for clarity

# Find the keypoints and compute the descriptors with ORB
keypoints_orb, descriptors_orb = orb.detectAndCompute(img, None)

print(f"ORB: Detected {len(keypoints_orb)} keypoints.")
if descriptors_orb is not None:
    print(f"ORB: Computed descriptors of shape {descriptors_orb.shape} (Num keypoints x Descriptor size in bytes)")
    print(f"ORB: Descriptor type: {descriptors_orb.dtype} (Binary)")
else:
    print("ORB: No descriptors computed.")

# Draw keypoints
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size and orientation of the keypoint are drawn
img_orb_keypoints = cv2.drawKeypoints(img_color_for_drawing, keypoints_orb, None, color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow('ORB Keypoints', img_orb_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --- Note on SIFT/SURF (Commented Out) ---
print("\n--- Note on SIFT/SURF --- ")
print("SIFT and SURF are powerful but potentially require `opencv-contrib-python`.")
print("They might also be subject to patents depending on usage and OpenCV version.")
# try:
#     # Example: SIFT (Requires opencv-contrib-python)
#     sift = cv2.xfeatures2d.SIFT_create()
#     keypoints_sift, descriptors_sift = sift.detectAndCompute(img, None)
#     print(f"SIFT: Detected {len(keypoints_sift)} keypoints. Descriptor shape: {descriptors_sift.shape}")
#     img_sift_keypoints = cv2.drawKeypoints(img_color_for_drawing, keypoints_sift, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     # cv2.imshow('SIFT Keypoints', img_sift_keypoints)
# except AttributeError:
#      print("cv2.xfeatures2d unavailable. Install opencv-contrib-python for SIFT/SURF.")
# except Exception as e:
#     print(f"Error using SIFT: {e}")
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --- Displaying Side-by-Side (using Matplotlib) --- #
try:
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1), plt.imshow(img_harris), plt.title('Harris Corners')
    plt.subplot(1, 3, 2), plt.imshow(img_shi_tomasi), plt.title('Shi-Tomasi Corners')
    plt.subplot(1, 3, 3), plt.imshow(img_orb_keypoints), plt.title('ORB Keypoints')
    plt.suptitle('Feature Detection Examples')
    # plt.show() # Uncomment to display plot
    print("\nPlots generated (call plt.show() to display).")
except NameError:
     print("\nSkipping plot display (likely OpenCV/plotting errors occurred earlier).")
except Exception as e:
     print(f"\nMatplotlib plotting failed: {e}")

print("\nScript finished.")
print("Image display/plotting is commented out. Uncomment relevant lines to view results.") 