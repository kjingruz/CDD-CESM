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
            {"id": 0, "name": "Benign"},
            {"id": 1, "name": "Malignant"},
            {"id": 2, "name": "Normal"},
        ],
    }
    
    for idx, row in df.iterrows():
        image_id = idx
        filename = row['file_name']
        width = row['width']
        height = row['height']
        bbox = json.loads(row['bbox'])
        
        coco_output["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })
        
        coco_output["annotations"].append({
            "id": idx,
            "image_id": image_id,
            "category_id": row["category_id"],
            "bbox": bbox,
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": [],
            "iscrowd": row["iscrowd"],
            "area": row["area"]
        })
    
    with open(output_file, 'w') as f:
        json.dump(coco_output, f, indent=4)
        
    print(f"COCO JSON file saved at: {output_file}")

if __name__ == "__main__":
    base_dir = '../../data'
    subsets = ['train', 'valid', 'test']
    
    for subset in subsets:
        csv_file = os.path.join(base_dir, f'{subset}_annotations.csv')
        image_dir = os.path.join(base_dir, subset)
        output_file = os.path.join(base_dir, f'{subset}_annotations.json')
        csv_to_coco(csv_file, image_dir, output_file)
