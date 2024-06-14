import cv2
import numpy as np
import os

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(otsu_thresh)
        cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_image = masked_image[y:y+h, x:x+w]
    else:
        cropped_image = image

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized_image = clahe.apply(cropped_image)
    
    return equalized_image

def preprocess_and_save(image_path, save_path):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        cv2.imwrite(save_path, preprocessed_image)
    else:
        print(f"Skipping image: {image_path} due to preprocessing failure")
