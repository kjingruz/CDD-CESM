import csv
import json
import pandas as pd

# Load segmentation annotations
segmentation_file = './data/Radiology_hand_drawn_segmentations_v2.csv'
annotations = []
with open(segmentation_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['region_shape_attributes']:  # Only process rows with region_shape_attributes
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

annotated_images = list(annotations_by_filename.keys())

# Load classification annotations
annotations_file = './data/Radiology-manual-annotations.xlsx'
df_annotations = pd.read_excel(annotations_file)

def classify_images(df):
    classifications = {}
    for index, row in df.iterrows():
        image_name = os.path.basename(row['Image_name'])  # Remove the path from the image name
        classification = row['Pathology Classification/ Follow up']
        
        if classification == 'Benign':
            classifications[image_name] = 0  # Use integer labels
        elif classification == 'Malignant':
            classifications[image_name] = 1
        else:
            classifications[image_name] = 2
        
        print(f"Classify: {image_name} as {classification} ({classifications[image_name]})")
    
    return classifications

classifications = classify_images(df_annotations)
