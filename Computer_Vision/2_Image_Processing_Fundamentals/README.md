# Topic 2: Image Processing Fundamentals

## Overview

Before diving into complex vision tasks like object recognition, it's essential to understand basic image processing techniques. These techniques manipulate pixel values directly to enhance images, extract simple features, or prepare them for further analysis. We'll primarily focus on operations applicable to grayscale images for simplicity, but many concepts extend to color images. Common libraries for these tasks include OpenCV (`cv2`), Pillow (`PIL`), and Scikit-image.

## 1. Point Operations

Point operations modify the value of each pixel independently of its neighbors. The transformation function `g(x,y) = T(f(x,y))` depends only on the value of the pixel `f(x,y)` itself.

*   **Brightness Adjustment:** Adds or subtracts a constant value to/from each pixel.
    *   `g(x,y) = f(x,y) + b` (where `b` controls brightness)
    *   Values are typically clipped to the valid range (e.g., 0-255 for uint8).
*   **Contrast Adjustment:** Multiplies each pixel value by a constant factor (gain).
    *   `g(x,y) = a * f(x,y)` (where `a > 1` increases contrast, `0 < a < 1` decreases contrast)
    *   Often combined with brightness adjustment: `g(x,y) = a * f(x,y) + b`.
*   **Thresholding:** Converts a grayscale image into a binary image based on a threshold value `T`.
    *   `g(x,y) = max_value` if `f(x,y) > T`
    *   `g(x,y) = 0` otherwise
    *   Useful for separating objects from the background when there's good contrast. Variations include binary inverse, truncate, to zero, etc. (See `cv2.threshold`).

## 2. Histograms

An image histogram represents the distribution of pixel intensity values in an image.

*   **Calculation:** Counts the number of pixels for each intensity value (0-255 for 8-bit grayscale).
*   **Visualization:** A plot with intensity values on the x-axis and pixel counts on the y-axis.
*   **Interpretation:** Provides insights into the image's brightness, contrast, and overall tonal range. A narrow histogram indicates low contrast, while a histogram skewed left/right indicates a dark/bright image.
*   **Histogram Equalization:** A technique that automatically enhances contrast by spreading out the most frequent intensity values. It attempts to flatten the histogram, resulting in a more uniform distribution of pixel intensities. (See `cv2.equalizeHist`).

## 3. Spatial Filtering (Convolution)

Spatial filtering operates on a neighborhood of pixels. A filter (or kernel) – a small matrix – slides across the image, and the output pixel value is calculated based on a weighted sum of the neighboring pixel values defined by the kernel. This process is called **convolution**.

*   **Kernel:** A small matrix (e.g., 3x3, 5x5) defining the weights applied to the neighborhood. The center of the kernel aligns with the current pixel being processed.
*   **Process:**
    1.  Align the kernel center over the pixel `(x,y)`.
    2.  Multiply each kernel element by the corresponding underlying image pixel value.
    3.  Sum up all the products.
    4.  Assign the sum to the output image pixel at `(x,y)`.
    5.  Repeat for all pixels (handling image borders is necessary - padding).
*   **Key Applications:**
    *   **Smoothing (Blurring):** Reduces noise and detail.
        *   *Box Filter (Averaging):* Kernel with equal weights (e.g., `[[1,1,1],[1,1,1],[1,1,1]] / 9`). (See `cv2.blur`, `cv2.boxFilter`).
        *   *Gaussian Filter:* Kernel weights follow a Gaussian distribution, giving more weight to central pixels. Very effective for noise reduction while preserving edges better than box filter. (See `cv2.GaussianBlur`).
        *   *Median Filter:* Replaces the central pixel with the median value of its neighborhood. Effective against salt-and-pepper noise. (Not strictly linear convolution, but a common spatial filter). (See `cv2.medianBlur`).
    *   **Sharpening:** Enhances edges and details. Kernels often have a positive center value and negative surrounding values (e.g., `[[0,-1,0],[-1,5,-1],[0,-1,0]]`).

## 4. Edge Detection

Edges represent significant changes in intensity and often correspond to object boundaries.

*   **Concept:** Detect sharp changes in pixel intensity using derivative approximations. High derivative values indicate edges.
*   **Gradient:** The gradient points in the direction of the greatest intensity change. Its magnitude indicates the strength of the edge.
*   **Common Techniques:**
    *   **Sobel Operator:** Uses two kernels (one for horizontal changes, one for vertical changes) to approximate the first derivative. Combines the results to find gradient magnitude and direction. (See `cv2.Sobel`).
    *   **Laplacian Operator:** Approximates the second derivative. Detects edges by finding zero-crossings. Sensitive to noise. (See `cv2.Laplacian`).
    *   **Canny Edge Detector:** A multi-stage algorithm considered state-of-the-art for many applications:
        1.  *Noise Reduction:* Apply Gaussian blur.
        2.  *Gradient Calculation:* Use Sobel filters to find intensity gradients.
        3.  *Non-maximum Suppression:* Thin edges by keeping only the local maxima in the gradient direction.
        4.  *Hysteresis Thresholding:* Use two thresholds (minVal and maxVal). Pixels above maxVal are definite edges. Pixels between minVal and maxVal are edges only if connected to definite edges. Pixels below minVal are discarded. (See `cv2.Canny`).

## Next Steps

With these fundamental image processing tools, we can now move on to **Feature Detection & Description**, where we'll learn techniques to identify key points and regions within an image that are robust to various transformations. 