import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import torch

torch.multiprocessing.set_sharing_strategy('file_system')

def load_dataset_from_csv(csv_file, image_dir):
    df = pd.read_csv(csv_file)
    dataset_dicts = []
    for idx, row in df.iterrows():
        record = {}
        filename = os.path.join(image_dir, row['file_name'])
        if not os.path.exists(filename):
            print(f"File {filename} does not exist, skipping.")
            continue
        height, width = row['height'], row['width']
        record["file_name"] = filename
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
    cfg.MODEL.WEIGHTS = os.path.join("/path/to/output/directory", "model_final.pth")
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")

    evaluation_results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(evaluation_results)

if __name__ == "__main__":
    register_dataset("my_dataset_val", "./data/annotations.csv", "./data/valid")
    main()
