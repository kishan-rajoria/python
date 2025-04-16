# Ensure required libraries are installed:
# pip install opencv-python numpy

import cv2
import numpy as np

print(f"OpenCV version: {cv2.__version__}")

# --- 1. Create a Sample Image using NumPy ---
# Create a small 5x5 BGR image (OpenCV uses BGR order by default)
# Data type is uint8 (unsigned 8-bit integer), common for images (0-255)
height = 5
width = 5
# Create a NumPy array initialized with zeros
# Shape: (height, width, channels)
# Let's make it blue
dummy_image_bgr = np.zeros((height, width, 3), dtype=np.uint8)
dummy_image_bgr[:, :, 0] = 255 # Set Blue channel to max
dummy_image_bgr[1:4, 1:4, 1] = 180 # Add some green in the middle

print("\n--- Image Properties --- ")

# --- 2. Inspect Image Properties ---
# Get dimensions (shape)
# shape returns a tuple: (height, width, channels)
img_shape = dummy_image_bgr.shape
img_height = img_shape[0]
img_width = img_shape[1]

print(f"Image Dimensions (HxW): {img_height} x {img_width}")

# Check number of channels (e.g., 3 for BGR/RGB, 1 for Grayscale)
if len(img_shape) == 3:
    img_channels = img_shape[2]
    print(f"Number of Channels: {img_channels}")
else:
    print("Number of Channels: 1 (Grayscale)")

# Get data type of pixels
img_dtype = dummy_image_bgr.dtype
print(f"Pixel Data Type: {img_dtype}")

# Calculate total number of pixels
total_pixels = dummy_image_bgr.size
print(f"Total number of elements (pixels * channels): {total_pixels}")
print(f"Total number of pixels (H * W): {img_height * img_width}")


# --- 3. Accessing Pixel Values ---
print("\n--- Accessing Pixel Values --- ")
# Access the pixel at row 0, column 0
# Remember OpenCV uses (row, column) indexing, which is (y, x)
px_y, px_x = 0, 0
pixel_value_bgr = dummy_image_bgr[px_y, px_x]
print(f"Value of pixel at ({px_y}, {px_x}): {pixel_value_bgr} (BGR format)")

# Access the pixel in the green patch we made
px_y, px_x = 2, 2
pixel_value_bgr_middle = dummy_image_bgr[px_y, px_x]
print(f"Value of pixel at ({px_y}, {px_x}): {pixel_value_bgr_middle} (BGR format)")

# Access just the Blue channel value of the pixel at (0, 0)
blue_value = dummy_image_bgr[0, 0, 0]
print(f"Blue channel value at (0, 0): {blue_value}")

# --- 4. Color Space Conversion ---
print("\n--- Color Space Conversion --- ")
# Convert the BGR image to Grayscale
try:
    dummy_image_gray = cv2.cvtColor(dummy_image_bgr, cv2.COLOR_BGR2GRAY)
    print("Image converted to Grayscale successfully.")

    # Check properties of the grayscale image
    gray_shape = dummy_image_gray.shape
    print(f"Grayscale Image Dimensions (HxW): {gray_shape[0]} x {gray_shape[1]}")
    print(f"Grayscale Pixel Data Type: {dummy_image_gray.dtype}")

    # Access a grayscale pixel value
    gray_pixel_value = dummy_image_gray[0, 0]
    print(f"Grayscale value of pixel at (0, 0): {gray_pixel_value}")

except cv2.error as e:
    print(f"OpenCV error during color conversion: {e}")
except Exception as e:
    print(f"An unexpected error occurred during color conversion: {e}")


# --- Note on Loading/Saving (Commented Out) ---
# In a real scenario, you'd load an image from a file:
# image_path = 'path/to/your/image.jpg'
# loaded_image = cv2.imread(image_path)
# if loaded_image is None:
#     print(f"Error: Could not load image from {image_path}")
# else:
#     print(f"Successfully loaded image from {image_path} with shape {loaded_image.shape}")
#     # Displaying images usually requires additional setup (e.g., matplotlib or cv2.imshow)
#     # cv2.imshow("Loaded Image", loaded_image)
#     # cv2.waitKey(0) # Wait for a key press
#     # cv2.destroyAllWindows()

# And save an image:
# cv2.imwrite('output_image.png', dummy_image_gray)
# print("Dummy grayscale image saved as 'output_image.png' (if uncommented)")

print("\nScript finished.") 