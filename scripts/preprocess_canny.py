# scripts/preprocess_canny.py
import os
from preprocess import preprocess_with_canny

# Paths
input_dir = "data/train/"
output_dir = "data/processed_train/"
os.makedirs(output_dir, exist_ok=True)

# Apply Canny to all training images
train_image_path = input_dir + "train_cal.png"
output_image_path = output_dir + "train_cal_canny.png"

preprocess_with_canny(train_image_path, output_image_path)
print(f"Processed Canny image saved to: {output_image_path}")
