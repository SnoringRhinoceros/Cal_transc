import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image

# Load the trained model
model = load_model('models/edge_detection_model.h5')

# Load and preprocess test image
test_image_path = 'data/test/test_cal.png'
test_image = cv2.imread(test_image_path)
test_image_resized = preprocess_image(test_image)

# Predict edges
predicted_mask = model.predict(np.expand_dims(test_image_resized, axis=0))[0]

# Threshold for binary mask
thresholded_mask = (predicted_mask > 0.5).astype(np.uint8)

# Resize mask back to original size
original_size_mask = cv2.resize(thresholded_mask, (test_image.shape[1], test_image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Save the result
cv2.imwrite('data/test/basic_mask.png', original_size_mask * 255)


def post_process_mask(binary_mask):
    """
    Post-process the binary mask to extract clean lines using smoothing, contour detection,
    and Hough Line Transform.
    """
    # Step 1: Enhance the binary mask
    enhanced_mask = (binary_mask * 255).astype(np.uint8)
    cv2.imwrite('data/test/enhanced_mask_debug.png', enhanced_mask)

    # Step 2: Smooth the mask to reduce noise
    smoothed_mask = cv2.GaussianBlur(enhanced_mask, (5, 5), 0)
    cv2.imwrite('data/test/smoothed_mask_debug_fixed.png', smoothed_mask)

    # Step 3: Detect edges using Canny edge detection
    edges = cv2.Canny(smoothed_mask, 10, 50)
    cv2.imwrite('data/test/canny_edges_debug_fixed.png', edges)

    # Step 4: Apply Hough Line Transform to find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=10, maxLineGap=5)
    print("Detected lines:", lines)

    # Step 5: Create a blank mask for the lines
    line_mask = np.zeros_like(binary_mask)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 1, thickness=1)

    return line_mask


# Apply post-processing to the thresholded mask
processed_mask = post_process_mask(thresholded_mask)

# Resize mask back to the original size of the test image
original_size_processed_mask = cv2.resize(processed_mask, (test_image.shape[1], test_image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Save the result
cv2.imwrite('data/test/final_mask.png', original_size_processed_mask * 255)

# Optional: Overlay the lines on the original image
overlay = test_image.copy()
overlay[original_size_processed_mask == 1] = [0, 255, 0]  # Green lines
cv2.imwrite('data/test/overlayed_image.png', overlay)