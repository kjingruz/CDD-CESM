import os
import pandas as pd
import json
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch

def load_dataset_from_csv(csv_file, image_dir):
    df = pd.read_csv(csv_file)
    dataset_dicts = []
    category_id_mapping = {0: "benign", 1: "malignant", 2: "normal"}
    
    for idx, row in df.iterrows():
        record = {}
        subfolder = category_id_mapping[row['category_id']]
        filename = os.path.join(image_dir, subfolder, row['file_name'])
        abs_filename = os.path.abspath(filename)
        print(f"Checking file: {abs_filename}")
        if not os.path.exists(abs_filename):
            print(f"File {abs_filename} does not exist, skipping.")
            continue
        
        # Dynamically read image dimensions
        with Image.open(abs_filename) as img:
            width, height = img.size
        
        record["file_name"] = abs_filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        record["annotations"] = [{
            "bbox": json.loads(row["bbox"]),
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": json.loads(row["segmentation"]),
            "category_id": row["category_id"],
            "iscrowd": row["iscrowd"],
            "area": row["area"]
        }]
        dataset_dicts.append(record)
    return dataset_dicts

def register_dataset(name, csv_file, image_dir):
    DatasetCatalog.register(name, lambda: load_dataset_from_csv(csv_file, image_dir))
    MetadataCatalog.get(name).set(thing_classes=["Benign", "Malignant", "Normal"])

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join("../output/train_lowtime", "model_final.pth")
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "my_dataset_test")

    # Save predictions in COCO JSON format
    evaluation_results = inference_on_dataset(predictor.model, test_loader, evaluator)
    print(evaluation_results)
    
    # Assuming evaluator is COCOEvaluator and has an attribute _predictions for storing results
    with open('predictions.json', 'w') as f:
        json.dump(evaluator._predictions, f)

if __name__ == "__main__":
    annotations_test_csv = os.path.abspath("../../data/annotations_test.csv")
    test_image_dir = os.path.abspath("../../data/test")
    print(f"Annotations CSV path: {annotations_test_csv}")
    print(f"Test image directory path: {test_image_dir}")
    register_dataset("my_dataset_test", annotations_test_csv, test_image_dir)
    main()
