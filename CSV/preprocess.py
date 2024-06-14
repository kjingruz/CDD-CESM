import os
import cv2
import csv
import json
import numpy as np
import pandas as pd

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

def load_annotations(segmentation_file, classification_file):
    # Load segmentation annotations
    annotations = []
    with open(segmentation_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['region_shape_attributes']:
                region_shape_attributes = json.loads(row['region_shape_attributes'])
                annotations.append({
                    'filename': row['#filename'],
                    'points': {
                        'all_points_x': region_shape_attributes.get('all_points_x', []),
                        'all_points_y': region_shape_attributes.get('all_points_y', [])
                    }
                })

    annotations_by_filename = {}
    for annotation in annotations:
        if annotation['filename'] not in annotations_by_filename:
            annotations_by_filename[annotation['filename']] = []
        annotations_by_filename[annotation['filename']].append(annotation['points'])

    df_annotations = pd.read_excel(classification_file)

    return annotations_by_filename, df_annotations

def classify_images(df):
    classifications = {}
    for index, row in df.iterrows():
        image_name = os.path.basename(row['Image_name'])
        classification = row['Pathology Classification/ Follow up']
        if classification == 'Benign':
            classifications[image_name] = 0
        elif classification == 'Malignant':
            classifications[image_name] = 1
        else:
            classifications[image_name] = 2
    return classifications

def process_images(annotations_by_filename, classifications, df_annotations):
    os.makedirs('../data/annotated_images', exist_ok=True)
    images_info = []
    annotations_info = []
    category_id_mapping = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
    tracked_category_counts = {'Benign': 0, 'Malignant': 0, 'Normal': 0}
    excel_classifications = df_annotations.set_index('Image_name')['Pathology Classification/ Follow up'].to_dict()

    for filename, points in annotations_by_filename.items():
        image_path = f'../data/images/{filename}'
        annotated_image_path = f'../data/annotated_images/{filename}'
        if os.path.exists(image_path):
            classification = classifications.get(os.path.basename(os.path.splitext(filename)[0]), 2)
            preprocess_and_save(image_path, annotated_image_path)
            image = cv2.imread(annotated_image_path)
            if image is None:
                print(f"Failed to load annotated image: {annotated_image_path}")
                continue
            height, width = image.shape[:2]
            if points:
                for annotation in points:
                    all_points_x = annotation['all_points_x']
                    all_points_y = annotation['all_points_y']
                    if all_points_x and all_points_y:
                        segmentation = [list(np.array([all_points_x, all_points_y]).T.flatten())]
                        bbox = [int(min(all_points_x)), int(min(all_points_y)), int(max(all_points_x) - min(all_points_x)), int(max(all_points_y) - min(all_points_y))]
                        area = int(cv2.contourArea(np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)))

                        if len(segmentation[0]) > 0:
                            annotations_info.append({
                                "id": len(annotations_info) + 1,
                                "image_id": len(images_info) + 1,
                                "category_id": classification,
                                "segmentation": segmentation,
                                "area": area,
                                "bbox": bbox,
                                "iscrowd": 0
                            })
            else:
                annotations_info.append({
                    "id": len(annotations_info) + 1,
                    "image_id": len(images_info) + 1,
                    "category_id": classification,
                    "segmentation": [[]],
                    "area": 0,
                    "bbox": [0, 0, 0, 0],
                    "iscrowd": 0
                })
            images_info.append({
                "id": len(images_info) + 1,
                "file_name": filename,
                "width": int(width),
                "height": int(height)
            })
            tracked_category_counts[category_id_mapping[classification]] += 1

    all_images = [f for f in os.listdir('../data/images') if f.endswith(('.jpg', '.jpeg'))]
    for image_file in all_images:
        if image_file not in annotations_by_filename:
            image_path = f'../data/images/{image_file}'
            annotated_image_path = f'../data/annotated_images/{image_file}'
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                height, width = image.shape[:2]
                images_info.append({
                    "id": len(images_info) + 1,
                    "file_name": image_file,
                    "width": int(width),
                    "height": int(height)
                })
                annotations_info.append({
                    "id": len(annotations_info) + 1,
                    "image_id": len(images_info),
                    "category_id": 2,
                    "segmentation": [[]],
                    "area": 0,
                    "bbox": [0, 0, 0, 0],
                    "iscrowd": 0
                })
                tracked_category_counts['Normal'] += 1

    print("Tracked category counts:", tracked_category_counts)
    return images_info, annotations_info

def convert_to_native_types(data):
    if isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, (np.generic, np.ndarray)):
        return data.tolist() if isinstance(data, np.ndarray) else data.item()
    else:
        return data

def save_annotations_to_csv(images_info, annotations_info, output_csv_path):
    annotations_info = convert_to_native_types(annotations_info)
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['file_name', 'width', 'height', 'category_id', 'segmentation', 'bbox', 'area', 'iscrowd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for img_info, ann_info in zip(images_info, annotations_info):
            row = {
                'file_name': img_info['file_name'],
                'width': img_info['width'],
                'height': img_info['height'],
                'category_id': ann_info['category_id'],
                'segmentation': json.dumps(ann_info['segmentation']),
                'bbox': json.dumps(ann_info['bbox']),
                'area': ann_info['area'],
                'iscrowd': ann_info['iscrowd']
            }
            writer.writerow(row)
    print(f"Created annotations CSV at: {output_csv_path}")

if __name__ == "__main__":
    segmentation_file = '../../data/Radiology_hand_drawn_segmentations_v2.csv'
    classification_file = '../../data/Radiology-manual-annotations.xlsx'
    output_csv_path = '../../data/annotations.csv'

    annotations_by_filename, df_annotations = load_annotations(segmentation_file, classification_file)
    classifications = classify_images(df_annotations)
    images_info, annotations_info = process_images(annotations_by_filename, classifications, df_annotations)
    save_annotations_to_csv(images_info, annotations_info, output_csv_path)
