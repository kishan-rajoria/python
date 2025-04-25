# Ensure required libraries are installed:
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
from matplotlib import pyplot as plt

print(f"OpenCV version: {cv2.__version__}")

# --- 1. Create a Sample Color Image ---
# Create a 150x300 BGR image with distinct color patches
height = 150
width = 300
img_bgr = np.zeros((height, width, 3), dtype=np.uint8)

# Define some colors (BGR format)
blue = [255, 0, 0]
_green = [0, 255, 0]
red = [0, 0, 255]
yellow = [0, 255, 255]

# Fill regions
img_bgr[:, 0:width//3] = blue
img_bgr[:, width//3:2*width//3] = _green
img_bgr[:, 2*width//3:width] = red
# Add a yellow circle
cv2.circle(img_bgr, (width//2, height//2), 30, yellow, -1)

print("\n--- Sample Color Image Created ---")
print(f"Image shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")

# --- Displaying (Optional) ---
# def display_image(window_name, image):
#     cv2.imshow(window_name, image)
#     cv2.waitKey(0)
# display_image('Original Color Image', img_bgr)
# cv2.destroyAllWindows()

# --- 2. Color-Based Segmentation (HSV) ---
print("\n--- Color Segmentation in HSV --- ")

# Convert BGR to HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Define range for GREEN color in HSV
# Note: HSV ranges in OpenCV are H: 0-179, S: 0-255, V: 0-255
# Green is typically around H=60. Let's define a range.
lower_green = np.array([40, 50, 50]) # Lower bound (adjust sensitivity here)
upper_green = np.array([80, 255, 255]) # Upper bound

# Create a mask for green color
mask_green = cv2.inRange(img_hsv, lower_green, upper_green)
print(f"Green mask created (shape: {mask_green.shape})")

# Apply the mask to the original image (optional)
# result_green_segment = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_green)
# print(f"Green segment isolated using bitwise_and")

# Display the mask
# display_image('Green Mask (HSV)', mask_green)
# display_image('Green Segment', result_green_segment)
# cv2.destroyAllWindows()

# --- 3. Thresholding-Based Segmentation --- 
print("\n--- Thresholding Segmentation (Grayscale + Otsu) --- ")

# Convert original image to grayscale
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Apply Otsu's thresholding
ret_otsu, thresh_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(f"Otsu Threshold applied: Optimal threshold={ret_otsu}")
# Using THRESH_BINARY_INV often helps separate foreground objects if they are darker than background
# Depending on the image, THRESH_BINARY might be more appropriate

# Display thresholded image
# display_image('Otsu Threshold (Grayscale)', thresh_otsu)
# cv2.destroyAllWindows()

# --- 4. Contour Detection (based on thresholding) --- 
print("\n--- Contour Detection on Thresholded Image --- ")

# Find contours
# cv2.RETR_EXTERNAL retrieves only the outer contours
# cv2.CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments, leaving only their end points
contours, hierarchy = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} contours.")

# Draw contours on a copy of the original image
img_contours = img_bgr.copy()
cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 2) # Draw all contours in white

# Display contours
# display_image('Contours on Original', img_contours)
# cv2.destroyAllWindows()

# --- Displaying Results Side-by-Side (using Matplotlib) --- #
try:
    plt.figure(figsize=(15, 5))
    # Need to convert BGR to RGB for Matplotlib
    plt.subplot(1, 4, 1), plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(1, 4, 2), plt.imshow(mask_green, cmap='gray'), plt.title('Green Mask (HSV)')
    plt.subplot(1, 4, 3), plt.imshow(thresh_otsu, cmap='gray'), plt.title('Otsu Threshold')
    plt.subplot(1, 4, 4), plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB)), plt.title('Contours')
    plt.suptitle('Image Segmentation Examples')
    # plt.show() # Uncomment to display plot
    print("\nPlots generated (call plt.show() to display).")
except Exception as e:
     print(f"\nMatplotlib plotting failed: {e}")

print("\nScript finished.")
print("Image display/plotting is commented out. Uncomment relevant lines to view results.") 