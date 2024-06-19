import pandas as pd
import json
import os
from detectron2.structures import BoxMode

def csv_to_coco(csv_file, image_dir, output_file):
    df = pd.read_csv(csv_file)
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Benign"},
            {"id": 2, "name": "Malignant"},
            {"id": 3, "name": "Normal"},
        ],
    }

    category_id_mapping = {0: "benign", 1: "malignant", 2: "normal"}

    for idx, row in df.iterrows():
        image_id = idx
        subfolder = category_id_mapping[int(row["Classification"])]
        filename = os.path.join(subfolder, row['file_name'])
        width = row['width']
        height = row['height']
        bbox = json.loads(row['bbox'])
        
        coco_output["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })
        
        category_id = int(row["Classification"]) + 1
        coco_output["annotations"].append({
            "id": idx,
            "image_id": image_id,
            "category_id": category_id,  # Ensure category ID is incremented by 1
            "bbox": bbox,
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": [],
            "iscrowd": int(row["iscrowd"]),
            "area": float(row["area"])
        })
    
    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=4)
        
    print(f"COCO JSON file saved at: {output_file}")

if __name__ == "__main__":
    base_dir = '/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/data'
    subsets = ['train', 'valid', 'test']
    
    for subset in subsets:
        csv_file = os.path.join(base_dir, f'{subset}_annotations.csv')
        image_dir = os.path.join(base_dir, subset)
        output_file = os.path.join(base_dir, f'{subset}_annotations.json')
        csv_to_coco(csv_file, image_dir, output_file)
