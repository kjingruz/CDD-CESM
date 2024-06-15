import pandas as pd
import json
import os

def csv_to_coco(csv_file, image_dir, output_file):
    df = pd.read_csv(csv_file)
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "Benign"},
            {"id": 2, "name": "Malignant"},
            {"id": 3, "name": "Normal"}
        ]
    }

    for idx, row in df.iterrows():
        filename = os.path.join(image_dir, row['file_name'])
        if not os.path.exists(filename):
            print(f"File {filename} does not exist, skipping.")
            continue

        height, width = row['height'], row['width']
        image_info = {
            "file_name": row['file_name'],
            "height": height,
            "width": width,
            "id": idx
        }
        coco_format["images"].append(image_info)

        annotation_info = {
            "bbox": json.loads(row["bbox"]),
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": json.loads(row["segmentation"]),
            "category_id": row["category_id"],
            "iscrowd": row["iscrowd"],
            "area": row["area"],
            "image_id": idx,
            "id": idx
        }
        coco_format["annotations"].append(annotation_info)

    with open(output_file, 'w') as f:
        json.dump(coco_format, f)

    print(f"Annotations have been converted to COCO format and saved to {output_file}")

# Convert annotations.csv to COCO JSON format
csv_to_coco('./data/annotations.csv', './data/valid', './data/annotations.json')
