# Ensure required libraries are installed:
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
from matplotlib import pyplot as plt # For histogram plotting

print(f"OpenCV version: {cv2.__version__}")

# --- 1. Create a Sample Grayscale Image ---
# Create a 100x200 grayscale image with different regions
height = 100
width = 200
img = np.zeros((height, width), dtype=np.uint8)

# Add regions with different intensities
img[0:height, 0:width//3] = 50       # Dark region
img[0:height, width//3:2*width//3] = 150 # Gray region
img[0:height, 2*width//3:width] = 220   # Bright region

# Add some noise (salt & pepper simulation)
noise = np.zeros((height, width), dtype=np.uint8)
cv2.randu(noise, 0, 255)
salt = noise > 245
pepper = noise < 10
img[salt] = 255
img[pepper] = 0

print("\n--- Sample Grayscale Image Created (with noise) ---")
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# --- Note on Displaying Images (Commented Out) ---
# To display images during execution, you'd typically use:
# cv2.imshow('Original Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Or use matplotlib:
# plt.imshow(img, cmap='gray')
# plt.title('Original Image')
# plt.show()

# --- 2. Point Operations: Thresholding ---
print("\n--- Thresholding Examples --- ")

# a) Simple Binary Thresholding
threshold_value = 127
max_value = 255
ret_simple, thresh_simple = cv2.threshold(img, threshold_value, max_value, cv2.THRESH_BINARY)
print(f"Simple Thresholding: ret={ret_simple}, output shape={thresh_simple.shape}")
# cv2.imshow('Simple Threshold', thresh_simple)

# b) Otsu's Binarization (Automatic Threshold)
ret_otsu, thresh_otsu = cv2.threshold(img, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu's Thresholding: Optimal threshold={ret_otsu}, output shape={thresh_otsu.shape}")
# cv2.imshow('Otsu Threshold', thresh_otsu)

# c) Adaptive Thresholding (Gaussian)
# Useful for varying illumination (though less apparent in our simple example)
block_size = 11 # Size of the neighborhood area
C = 2 # Constant subtracted from the mean/weighted mean
thresh_adaptive = cv2.adaptiveThreshold(img, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block_size, C)
print(f"Adaptive Thresholding (Gaussian): output shape={thresh_adaptive.shape}")
# cv2.imshow('Adaptive Threshold', thresh_adaptive)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# --- 3. Histograms --- 
print("\n--- Histogram Examples --- ")

# a) Calculate Histogram
# Params: images, channels, mask, histSize, ranges
hist_original = cv2.calcHist([img], [0], None, [256], [0, 256])
print("Histogram calculated.")

# b) Plot Histogram (using Matplotlib)
try:
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist_original, color='black')
    plt.title('Original Histogram')
    plt.xlabel("Intensity")
    plt.ylabel("# Pixels")
    plt.xlim([0, 256])
except Exception as e:
    print(f"Matplotlib plotting failed: {e}")
    print("Install matplotlib (`pip install matplotlib`) to see plots.")

# c) Histogram Equalization
equalized_img = cv2.equalizeHist(img)
hist_equalized = cv2.calcHist([equalized_img], [0], None, [256], [0, 256])
print("Histogram equalization applied.")
# cv2.imshow('Equalized Image', equalized_img)

# Plot Equalized Histogram
try:
    plt.subplot(1, 2, 2)
    plt.plot(hist_equalized, color='black')
    plt.title('Equalized Histogram')
    plt.xlabel("Intensity")
    plt.ylabel("# Pixels")
    plt.xlim([0, 256])
    plt.tight_layout()
    # plt.show() # Uncomment to display plot during execution
    print("Histogram plots generated (call plt.show() to display).")
except Exception as e:
    print(f"Matplotlib plotting failed: {e}")

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# --- 4. Spatial Filtering (Smoothing/Blurring) --- 
print("\n--- Smoothing Filter Examples --- ")

# a) Gaussian Blur
# ksize must be odd
gaussian_blurred = cv2.GaussianBlur(img, (5, 5), 0)
print(f"Gaussian Blur applied (5x5 kernel): output shape={gaussian_blurred.shape}")
# cv2.imshow('Gaussian Blur', gaussian_blurred)

# b) Median Blur (Effective for salt-and-pepper noise)
# ksize must be odd and > 1
median_blurred = cv2.medianBlur(img, 5)
print(f"Median Blur applied (5x5 kernel): output shape={median_blurred.shape}")
# cv2.imshow('Median Blur', median_blurred)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --- 5. Edge Detection --- 
print("\n--- Edge Detection Examples --- ")
# Often beneficial to blur slightly before edge detection
img_blurred_for_edges = cv2.GaussianBlur(img, (3, 3), 0)

# a) Sobel Edge Detection (X and Y directions)
# cv2.CV_64F output depth allows for negative gradient values, then take absolute
sobelx = cv2.Sobel(img_blurred_for_edges, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_blurred_for_edges, cv2.CV_64F, 0, 1, ksize=3)
sobelx_abs = cv2.convertScaleAbs(sobelx) # Convert back to uint8
sobely_abs = cv2.convertScaleAbs(sobely)
sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)
print(f"Sobel Edge Detection applied: output shape={sobel_combined.shape}")
# cv2.imshow('Sobel X', sobelx_abs)
# cv2.imshow('Sobel Y', sobely_abs)
# cv2.imshow('Sobel Combined', sobel_combined)

# b) Canny Edge Detection
# Requires minVal and maxVal thresholds for hysteresis
canny_edges = cv2.Canny(img_blurred_for_edges, threshold1=50, threshold2=150)
print(f"Canny Edge Detection applied: output shape={canny_edges.shape}")
# cv2.imshow('Canny Edges', canny_edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print("\nScript finished.")
print("Image display/plotting is commented out. Uncomment cv2.imshow/plt.show lines to view results.") 