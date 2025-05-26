# Ensure required libraries are installed:
# pip install opencv-python numpy

import cv2
import numpy as np
import time

print(f"OpenCV version: {cv2.__version__}")

# --- Configuration --- 
# Option 1: Use a video file
# video_source = 'your_video.mp4' # Replace with the path to your video file

# Option 2: Use a connected camera (usually index 0)
video_source = 0

print("\n--- Basic Video Processing Demonstration --- ")

# --- 1. Open Video Source --- 
cap = cv2.VideoCapture(video_source)

# Check if the source was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video source: {video_source}")
    exit()

print(f"Successfully opened video source: {video_source}")

# Get video properties (optional, might not work for all camera streams initially)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
framerate = cap.get(cv2.CAP_PROP_FPS)
print(f"Video properties (if available): {width}x{height} @ {framerate:.2f} FPS")

# --- 2. Process Frames --- 
print("\nProcessing video frames... Press 'q' to quit.")

frame_count = 0
start_time = time.time()

while True:
    # a) Read a frame
    ret, frame = cap.read()

    # Check if frame was read successfully (ret is True)
    # If reading from file, ret becomes False at the end
    if not ret:
        if isinstance(video_source, str): # Check if it was a file
            print("\nEnd of video file reached.")
        else:
            print("\nError reading frame from camera.")
        break # Exit the loop

    # b) Perform simple processing on the frame
    # Example: Convert to grayscale
    processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Example: Add frame number text (optional)
    frame_count += 1
    cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # c) Display the original and processed frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Processed Frame (Grayscale)', processed_frame)

    # d) Check for user input to quit
    # waitKey(1) waits 1ms for a key press. Necessary for imshow to work.
    # 0xFF == ord('q') checks if the pressed key was 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n'q' pressed, exiting loop.")
        break

# --- 3. Release Resources --- 
end_time = time.time()
print("\nReleasing video capture object and closing windows...")
cap.release() # Release the video capture object
cv2.destroyAllWindows() # Close all OpenCV windows

# --- Summary --- 
duration = end_time - start_time
if duration > 0:
    actual_fps = frame_count / duration
    print(f"\nProcessed {frame_count} frames in {duration:.2f} seconds ({actual_fps:.2f} FPS).")
else:
    print(f"\nProcessed {frame_count} frames.")

print("Script finished.") 