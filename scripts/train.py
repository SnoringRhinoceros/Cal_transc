import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
from preprocess import parse_annotations, normalize_boxes, create_mask, preprocess_image
from model import build_model

# Load training data
train_image_path = 'data/train/train_cal.png'
annotation_path = 'data/train/pair.txt'

train_image = cv2.imread(train_image_path)
train_boxes = normalize_boxes(parse_annotations(annotation_path), train_image.shape)
train_mask = create_mask(train_image.shape, train_boxes)

# Preprocess training image and mask
train_image_resized = preprocess_image(train_image)
train_mask_resized = preprocess_image(train_mask, target_size=(512, 512))

# Build and train the model
model = build_model()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    np.expand_dims(train_image_resized, axis=0), 
    np.expand_dims(train_mask_resized, axis=0), 
    epochs=10, 
    batch_size=1
)

# Save the trained model
model.save('models/edge_detection_model.h5')
