# Ensure required libraries are installed:
# pip install opencv-python numpy matplotlib

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import requests # To potentially download the cascade file

print(f"OpenCV version: {cv2.__version__}")

# --- 1. Load Haar Cascade Classifier --- 
# Haar cascades are XML files containing pre-trained classifiers.
# OpenCV comes with several pre-trained cascades (faces, eyes, etc.).
# You need the path to the XML file.

# Option A: Specify path if you know where it is (common location within OpenCV installation)
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_cascade_path = os.path.join(cv2_base_dir, 'data', 'haarcascade_frontalface_default.xml')

# Option B: Download if not found (or specify a different path)
haar_cascade_filename = 'haarcascade_frontalface_default.xml'
if not os.path.exists(haar_cascade_path):
    print(f"Default path not found: {haar_cascade_path}")
    # Check current directory
    if os.path.exists(haar_cascade_filename):
        haar_cascade_path = haar_cascade_filename
        print(f"Found {haar_cascade_filename} in current directory.")
    else:
        print(f"Attempting to download {haar_cascade_filename}...")
        url = f'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{haar_cascade_filename}'
        try:
            r = requests.get(url, allow_redirects=True)
            r.raise_for_status() # Raise an exception for bad status codes
            with open(haar_cascade_filename, 'wb') as f:
                f.write(r.content)
            haar_cascade_path = haar_cascade_filename
            print(f"Successfully downloaded to {haar_cascade_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading cascade file: {e}")
            print("Please download 'haarcascade_frontalface_default.xml' manually")
            print("from the OpenCV GitHub repository and place it in the script's directory.")
            haar_cascade_path = None

face_cascade = None
if haar_cascade_path and os.path.exists(haar_cascade_path):
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        print(f"Error loading Haar Cascade from: {haar_cascade_path}")
        face_cascade = None # Mark as failed
    else:
        print(f"Successfully loaded Haar Cascade: {os.path.basename(haar_cascade_path)}")
else:
    print("Haar Cascade file not found or path not set. Face detection will be skipped.")

# --- 2. Load a Sample Image --- 
# For face detection, an image with faces is needed.
# We'll create a simple placeholder if no cascade is loaded, or try loading a common test image.

img_bgr = None
img_gray = None

# Try to load a standard test image if the cascade loaded
if face_cascade:
    # You might replace 'lena.png' with another image file containing faces
    # Or download one if needed
    test_image_filename = 'lena.png' # Placeholder - replace if needed
    if not os.path.exists(test_image_filename):
        print(f"{test_image_filename} not found. Creating a simple image instead.")
    else:
        img_bgr = cv2.imread(test_image_filename)
        if img_bgr is None:
             print(f"Error reading {test_image_filename}. Creating simple image.")

# Fallback: Create a simple image with shapes if cascade failed or image load failed
if img_bgr is None:
    print("Creating a simple image with shapes for demonstration.")
    height = 200
    width = 300
    img_bgr = np.ones((height, width, 3), dtype=np.uint8) * 255 # White background
    cv2.rectangle(img_bgr, (50, 50), (150, 150), (0, 255, 0), 3) # Green rectangle
    cv2.circle(img_bgr, (220, 100), 40, (0, 0, 255), -1)     # Red circle
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
else:
    print(f"Using image: {test_image_filename}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

print(f"Loaded/Created image shape: {img_bgr.shape}")

# --- 3. Object Detection (using Haar Cascade) --- 
print("\n--- Object Detection (Faces) --- ")

faces = []
if face_cascade and img_gray is not None:
    # Detect faces
    # detectMultiScale parameters:
    #   scaleFactor: How much the image size is reduced at each image scale.
    #   minNeighbors: How many neighbors each candidate rectangle should have to retain it.
    #   minSize: Minimum possible object size. Objects smaller than this are ignored.
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Detected {len(faces)} face(s)." if faces is not None else "Detected 0 faces.")
    if faces is None: faces = [] # Ensure faces is iterable
else:
    print("Skipping face detection (cascade not loaded or error).")

# --- 4. Draw Bounding Boxes --- 
print("\n--- Drawing Bounding Boxes --- ")
img_detections = img_bgr.copy()

if len(faces) > 0:
    for (x, y, w, h) in faces:
        cv2.rectangle(img_detections, (x, y), (x+w, y+h), (255, 0, 0), 2) # Draw blue rectangle
    print(f"Drew {len(faces)} bounding box(es).")
else:
    print("No faces to draw bounding boxes for.")
    if face_cascade is None:
        print("(Haar cascade was not loaded)")

# --- Displaying Results (using Matplotlib) --- #
try:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_detections, cv2.COLOR_BGR2RGB))
    plt.title(f'Detected Objects ({len(faces)} faces found)' if face_cascade else 'Original Image (No Detection)')
    plt.axis('off')

    plt.suptitle('Haar Cascade Object Detection Example')
    # plt.show() # Uncomment to display plot
    print("\nPlots generated (call plt.show() to display).")
except Exception as e:
     print(f"\nMatplotlib plotting failed: {e}")

print("\nScript finished.")
print("Ensure you have the Haar cascade XML file (e.g., 'haarcascade_frontalface_default.xml').")
print("If using a real image, place it in the same directory or provide the full path.")
print("Image display/plotting is commented out. Uncomment relevant lines to view results.") 