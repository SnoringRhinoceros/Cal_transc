import numpy as np
import cv2
import ast

def parse_annotations(filepath):
    """
    Parse the annotation file to extract bounding box coordinates.
    """
    annotations = []
    with open(filepath, 'r') as f:
        for line in f:
            array_version = ast.literal_eval(line)
            top_left = array_version[2]
            bottom_right = array_version[3]
            x1, y1 = map(float, top_left)
            x2, y2 = map(float, bottom_right)
            annotations.append((x1, y1, x2, y2))
            
    return np.array(annotations)

def normalize_boxes(boxes, image_shape):
    """
    Normalize bounding box coordinates relative to the image dimensions.
    """
    h, w = image_shape[:2]
    return boxes / np.array([w, h, w, h])

def create_mask(image_shape, boxes):
    """
    Create a binary mask from bounding boxes.
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        mask[y1:y2, x1:x2] = 1
    return mask

def preprocess_image(image, target_size=(512, 512)):
    """
    Resize an image to the target size.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)