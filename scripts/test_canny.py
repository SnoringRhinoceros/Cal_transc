import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image, apply_canny

# Load the trained model
model = load_model('models/edge_detection_model_canny.h5')

# Load and preprocess test image with Canny
test_image_path = 'data/test/test_cal.png'
test_image = cv2.imread(test_image_path)
test_image_canny = apply_canny(test_image)
test_image_resized = preprocess_image(test_image_canny)

# Predict edges
predicted_mask = model.predict(np.expand_dims(test_image_resized, axis=(0, -1)))[0]

# Threshold for binary mask
thresholded_mask = (predicted_mask > 0.5).astype(np.uint8)

# Resize mask back to the original size of the test image
original_size_mask = cv2.resize(thresholded_mask, (test_image.shape[1], test_image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Save the result
cv2.imwrite('data/test/predicted_mask_canny.png', original_size_mask * 255)
print("Predicted mask saved to ../data/test/predicted_mask_canny.png")
