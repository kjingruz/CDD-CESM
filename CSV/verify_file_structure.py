import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2
import json
import numpy as np
import csv

def create_directories(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def extract_patient_id(filename):
    return filename.split('_')[0][1:]

def save_subset_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} records to {file_path}")

def add_columns(df, image_dir):
    # Add necessary columns with placeholder values
    df['width'] = 0
    df['height'] = 0
    df['bbox'] = '[]'
    df['iscrowd'] = 0
    df['area'] = 0

    for idx, row in df.iterrows():
        image_path = os.path.join(image_dir, row['file_name'])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            df.at[idx, 'width'] = width
            df.at[idx, 'height'] = height
            # Assuming bbox covers the entire image for now
            df.at[idx, 'bbox'] = json.dumps([0, 0, width, height])
            df.at[idx, 'area'] = width * height
        else:
            print(f"Image not found: {image_path}")

    return df

def verify_and_copy_files(excel_file, image_dir, train_dir, val_dir, test_dir, annotations, train_ratio=0.73, val_ratio=0.12, test_ratio=0.15):
    # Check if Excel file exists
    if not os.path.exists(excel_file):
        print(f"Excel file {excel_file} does not exist.")
        return False

    # Load Excel file
    try:
        df = pd.read_excel(excel_file, sheet_name=0)  # Ensure correct sheet name
    except Exception as e:
        print(f"Failed to load Excel file: {e}")
        return False

    # Ensure the dataframe has the required columns
    required_columns = ['Image_name', 'Pathology Classification/ Follow up']
    if not all(col in df.columns for col in required_columns):
        print("Excel file must contain 'Image_name' and 'Pathology Classification/ Follow up' columns.")
        return False

    # Rename the classification column for consistency
    df.rename(columns={'Pathology Classification/ Follow up': 'category_id', 'Image_name': 'file_name'}, inplace=True)

    # Normalize file extensions and remove spaces
    df['file_name'] = df['file_name'].apply(lambda x: ' '.join(x.split()).strip() + '.jpg')

    # Extract patient ID and add it to the dataframe
    df['patient_id'] = df['file_name'].apply(extract_patient_id)

    # Group by patient_id
    grouped = df.groupby('patient_id')

    # Create a list of groups (each group corresponds to one patient)
    patient_groups = [group for _, group in grouped]

    # Split the patient groups into training, validation, and test sets
    train_groups, temp_groups = train_test_split(patient_groups, train_size=train_ratio, random_state=42)
    val_groups, test_groups = train_test_split(temp_groups, train_size=val_ratio/(val_ratio + test_ratio), random_state=42)

    train_subset = pd.concat(train_groups)
    val_subset = pd.concat(val_groups)
    test_subset = pd.concat(test_groups)

    # Add columns to each subset
    train_subset = add_columns(train_subset, image_dir)
    val_subset = add_columns(val_subset, image_dir)
    test_subset = add_columns(test_subset, image_dir)

    def copy_files(groups, target_dir, subgroup):
        all_files_exist = True
        missing_files = []
        for group in groups:
            for _, row in group.iterrows():
                filename = row['file_name']
                category_id = row['category_id']
                if category_id == 'Benign':
                    classification = 'benign'
                elif category_id == 'Malignant':
                    classification = 'malignant'
                else:
                    classification = 'normal'

                src_path = os.path.join(image_dir, filename)
                dst_dir = os.path.join(target_dir, classification)
                dst_path = os.path.join(dst_dir, filename)

                if not os.path.exists(src_path):
                    print(f"File {src_path} does not exist, skipping.")
                    missing_files.append(src_path)
                    all_files_exist = False
                else:
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    shutil.copy(src_path, dst_path)
                    print(f"Copied {src_path} to {dst_path}")
                    # Update annotations dataframe with subgroup
                    annotations.loc[annotations['file_name'] == filename, 'subgroup'] = subgroup

        if missing_files:
            print("Missing files:")
            for file in missing_files:
                print(file)

        return all_files_exist, missing_files

    # Verify and copy training files
    print("Copying training files...")
    train_files_exist, missing_train_files = copy_files(train_groups, train_dir, 'train')
    # Verify and copy validation files
    print("Copying validation files...")
    val_files_exist, missing_val_files = copy_files(val_groups, val_dir, 'valid')
    # Verify and copy test files
    print("Copying test files...")
    test_files_exist, missing_test_files = copy_files(test_groups, test_dir, 'test')

    # Attempt to copy missing files again
    if not train_files_exist or not val_files_exist or not test_files_exist:
        print("Re-attempting to copy missing files...")
        for missing_file in missing_train_files:
            src_path = os.path.join(image_dir, os.path.basename(missing_file))
            category_id = df[df['file_name'] == os.path.basename(missing_file)]['category_id'].values[0]
            if category_id == 'Benign':
                classification = 'benign'
            elif category_id == 'Malignant':
                classification = 'malignant'
            else:
                classification = 'normal'
            dst_dir = os.path.join(train_dir, classification)
            dst_path = os.path.join(dst_dir, os.path.basename(missing_file))
            if os.path.exists(src_path):
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy(src_path, dst_path)
                print(f"Re-attempted copy: {src_path} to {dst_path}")

        for missing_file in missing_val_files:
            src_path = os.path.join(image_dir, os.path.basename(missing_file))
            category_id = df[df['file_name'] == os.path.basename(missing_file)]['category_id'].values[0]
            if category_id == 'Benign':
                classification = 'benign'
            elif category_id == 'Malignant':
                classification = 'malignant'
            else:
                classification = 'normal'
            dst_dir = os.path.join(val_dir, classification)
            dst_path = os.path.join(dst_dir, os.path.basename(missing_file))
            if os.path.exists(src_path):
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy(src_path, dst_path)
                print(f"Re-attempted copy: {src_path} to {dst_path}")

        for missing_file in missing_test_files:
            src_path = os.path.join(image_dir, os.path.basename(missing_file))
            category_id = df[df['file_name'] == os.path.basename(missing_file)]['category_id'].values[0]
            if category_id == 'Benign':
                classification = 'benign'
            elif category_id == 'Malignant':
                classification = 'malignant'
            else:
                classification = 'normal'
            dst_dir = os.path.join(test_dir, classification)
            dst_path = os.path.join(dst_dir, os.path.basename(missing_file))
            if os.path.exists(src_path):
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                shutil.copy(src_path, dst_path)
                print(f"Re-attempted copy: {src_path} to {dst_path}")

    # Save the subsets to separate CSV files
    save_subset_to_csv(train_subset, '../../data/train_annotations.csv')
    save_subset_to_csv(val_subset, '../../data/valid_annotations.csv')
    save_subset_to_csv(test_subset, '../../data/test_annotations.csv')

    return train_files_exist and val_files_exist and test_files_exist

def verify_classifications(excel_file):
    # Load Excel file
    try:
        df = pd.read_excel(excel_file, sheet_name=0)  # Ensure correct sheet name
    except Exception as e:
        print(f"Failed to load Excel file: {e}")
        return False

    # Ensure the dataframe has the required columns
    required_columns = ['Image_name', 'Pathology Classification/ Follow up']
    if not all(col in df.columns for col in required_columns):
        print("Excel file must contain 'Image_name' and 'Pathology Classification/ Follow up' columns.")
        return False

    # Rename the classification column for consistency
    df.rename(columns={'Pathology Classification/ Follow up': 'category_id', 'Image_name': 'file_name'}, inplace=True)

    # Normalize filenames to remove leading/trailing spaces and multiple spaces
    df['file_name'] = df['file_name'].apply(lambda x: ' '.join(x.split()).strip() + '.jpg')

        # Verify classifications
    for _, row in df.iterrows():
        filename = row['file_name']
        classification = row['category_id']
        if classification not in ['Benign', 'Malignant', 'Normal']:
            print(f"Invalid classification {classification} for file {filename}")
            return False

    print("All classifications are verified.")
    return True

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

    df_annotations = pd.read_excel(classification_file, sheet_name=0)  # Ensure correct sheet name

    print("Columns in the dataframe:", df_annotations.columns)
    return annotations_by_filename, df_annotations

def classify_images(df):
    classifications = {}
    for index, row in df.iterrows():
        image_name = os.path.basename(row['Image_name'])  # Ensure 'Image_name' matches the column in the Excel file
        classification = row['Classification']
        if classification == 'Benign':
            classifications[image_name] = 0
        elif classification == 'Malignant':
            classifications[image_name] = 1
        else:
            classifications[image_name] = 2
    return classifications

def process_images(annotations_by_filename, classifications, df_annotations):
    os.makedirs('../../data/annotated_images', exist_ok=True)
    images_info = []
    annotations_info = []
    category_id_mapping = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}
    tracked_category_counts = {'Benign': 0, 'Malignant': 0, 'Normal': 0}
    failed_images = []

    for filename, points in annotations_by_filename.items():
        image_path = f'../../data/images/{filename}'
        annotated_image_path = f'../../data/annotated_images/{filename}'
        if os.path.exists(image_path):
            classification = classifications.get(os.path.basename(os.path.splitext(filename)[0]), 2)
            preprocess_and_save(image_path, annotated_image_path)
            image = cv2.imread(annotated_image_path)
            if image is None:
                print(f"Failed to load annotated image: {annotated_image_path}")
                failed_images.append(annotated_image_path)
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

    all_images = [f for f in os.listdir('../../data/images') if f.endswith(('.jpg', '.jpeg'))]
    for image_file in all_images:
        if image_file not in annotations_by_filename:
            image_path = f'../../data/images/{image_file}'
            annotated_image_path = f'../../data/annotated_images/{image_file}'
            if os.path.exists(image_path):
                preprocess_and_save(image_path, annotated_image_path)
                image = cv2.imread(annotated_image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    failed_images.append(image_path)
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
    print(f"Failed to process {len(failed_images)} images.")
    if failed_images:
        print("Failed images:")
        for img in failed_images:
            print(img)
    
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
        fieldnames = ['file_name', 'width', 'height', 'category_id', 'segmentation', 'bbox', 'area', 'iscrowd', 'subgroup']
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
                'iscrowd': ann_info['iscrowd'],
                'subgroup': img_info.get('subgroup', '')
            }
            writer.writerow(row)
    print(f"Created annotations CSV at: {output_csv_path}")

def compare_annotations(annotation_csv, classification_file):
    annotation_df = pd.read_csv(annotation_csv)
    classification_df = pd.read_excel(classification_file, sheet_name=0)
    classification_df.rename(columns={'Pathology Classification/ Follow up': 'classification', 'Image_name': 'file_name'}, inplace=True)

    discrepancies = []

    for _, row in annotation_df.iterrows():
        file_name = row['file_name']
        csv_classification = row['category_id']
        original_classification = classification_df[classification_df['file_name'] == file_name]['classification'].values

        if len(original_classification) > 0:
            original_classification = original_classification[0]
            if original_classification == 'Benign':
                original_classification_id = 0
            elif original_classification == 'Malignant':
                original_classification_id = 1
            else:
                original_classification_id = 2

            if csv_classification != original_classification_id:
                discrepancies.append(file_name)

    return discrepancies

# Adding this line to define the annotations DataFrame
annotations = pd.DataFrame(columns=['file_name', 'width', 'height', 'category_id', 'segmentation', 'bbox', 'area', 'iscrowd', 'subgroup'])

if __name__ == "__main__":
    segmentation_file = '../../data/Radiology_hand_drawn_segmentations_v2.csv'
    classification_file = '../../data/Radiology-manual-annotations.xlsx'
    output_csv_path = '../../data/annotations.csv'
    annotation_csv = '../../data/annotations.csv'

    annotations_by_filename, df_annotations = load_annotations(segmentation_file, classification_file)
    classifications = classify_images(df_annotations)
    images_info, annotations_info = process_images(annotations_by_filename, classifications, df_annotations)
    save_annotations_to_csv(images_info, annotations_info, output_csv_path)

    discrepancies = compare_annotations(annotation_csv, classification_file)
    if discrepancies:
        print("Discrepancies found in the following files:")
        for file in discrepancies:
            print(file)
    else:
        print("No discrepancies found.")

    image_dir = '../../data/images'
    train_dir = '../../data/train'
    val_dir = '../../data/val'
    test_dir = '../../data/test'

    # Create directories if they don't exist
    create_directories([train_dir, val_dir, test_dir])

    # Verify classifications
    if verify_classifications(classification_file):
        # Verify files and copy them to the correct directories
        if verify_and_copy_files(classification_file, image_dir, train_dir, val_dir, test_dir, annotations):
            print("All files are correctly set up.")
        else:
            print("There are missing files or directories.")
    else:
        print("There are issues with the classifications.")


