import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
from preprocess import preprocess_image, parse_annotations, normalize_boxes, create_mask
from model import build_model

# Load Canny-processed training data
train_image_path = 'data/processed_train/train_cal_canny.png'
annotation_path = 'data/train/pair.txt'

# Preprocess images and masks
train_image = cv2.imread(train_image_path, cv2.IMREAD_GRAYSCALE)
train_image_resized = preprocess_image(train_image)

train_boxes = normalize_boxes(parse_annotations(annotation_path), train_image.shape)
train_mask = create_mask(train_image.shape, train_boxes)
train_mask_resized = preprocess_image(train_mask, target_size=(512, 512))

# Add channel dimensions
train_image_resized = np.expand_dims(train_image_resized, axis=-1)
train_mask_resized = np.expand_dims(train_mask_resized, axis=-1)

# Validate shapes
print("Train image shape:", train_image_resized.shape)  # Should be (512, 512, 1)
print("Train mask shape:", train_mask_resized.shape)  # Should be (512, 512, 1)

# Build and train the model
model = build_model(input_shape=(512, 512, 1))
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    np.expand_dims(train_image_resized, axis=0),  # Add batch dimension
    np.expand_dims(train_mask_resized, axis=0),  # Add batch dimension
    epochs=10,
    batch_size=1
)

# Save the trained model
model.save('models/edge_detection_model_canny.h5')
print("Model trained and saved to models/edge_detection_model_canny.h5")
