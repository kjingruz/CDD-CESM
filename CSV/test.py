import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import torch

def register_dataset(name, json_file, image_dir):
    DatasetCatalog.register(name, lambda: json.load(open(json_file)))
    MetadataCatalog.get(name).set(thing_classes=["Benign", "Malignant", "Normal"])

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join("../output/train_lowtime/2nd_Attempt", "model_final.pth")
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
    annotations_test_json = os.path.abspath("../../data/test_annotations.json")
    test_image_dir = os.path.abspath("../../data/test")
    print(f"Annotations JSON path: {annotations_test_json}")
    print(f"Test image directory path: {test_image_dir}")
    register_dataset("my_dataset_test", annotations_test_json, test_image_dir)
    main()
