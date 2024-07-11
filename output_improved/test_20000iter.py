import torch
import os
import json
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer


def setup_cfg(weights_file):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = weights_file
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TEST = ("cesm_test",)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def load_coco_json(json_file, image_root):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    dataset_dicts = []
    for image in coco_data['images']:
        record = {}
        record["file_name"] = os.path.join(image_root, image["file_name"])
        record["height"] = image["height"]
        record["width"] = image["width"]
        record["image_id"] = image["id"]
        
        annos = [anno for anno in coco_data['annotations'] if anno['image_id'] == image['id']]
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno["category_id"],
                "iscrowd": anno.get("iscrowd", 0)
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

def generate_predictions_coco(cfg, model, dataset_name):
    evaluator = COCOEvaluator(dataset_name, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(model, val_loader, evaluator)
    
    coco_results = []
    if "bbox" in results:
        for img_id, img_results in results["bbox"].items():
            if isinstance(img_results, dict):
                prediction = {
                    "image_id": int(img_id),
                    "category_id": int(img_results.get("category_id", 0)),
                    "bbox": img_results.get("bbox", []),
                    "score": float(img_results.get("score", 0))
                }
                coco_results.append(prediction)
            elif isinstance(img_results, list):
                for pred in img_results:
                    prediction = {
                        "image_id": int(img_id),
                        "category_id": int(pred.get("category_id", 0)),
                        "bbox": pred.get("bbox", []),
                        "score": float(pred.get("score", 0))
                    }
                    coco_results.append(prediction)
    
    with open("predictions_coco.json", "w") as f:
        json.dump(coco_results, f)

    print("Predictions saved in COCO format to predictions_coco.json")
    return coco_results

def visualize_predictions(predictor, dataset_dicts, num_images=20):
    metadata = MetadataCatalog.get("cesm_test")
    class_names = metadata.thing_classes
    colors = {"gt": (0, 255, 0), "pred": (255, 255, 0)}  # Green for GT, Bright Yellow for predictions

    os.makedirs("output_images", exist_ok=True)
    
    for i, d in enumerate(random.sample(dataset_dicts, num_images)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        
        # Create two copies of the image for GT and predictions
        im_gt = im.copy()
        im_pred = im.copy()
        
        # Draw ground truth boxes and labels
        gt_class = None
        for ann in d["annotations"]:
            box = ann["bbox"]
            im_gt = cv2.rectangle(im_gt, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), colors["gt"], 2)
            class_id = ann["category_id"]
            gt_class = class_names[class_id]
        
        cv2.putText(im_gt, f"Ground Truth: {gt_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors["gt"], 2)
        
        # Draw predicted boxes and labels
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        max_score = 0
        pred_class = None
        for box, score, class_id in zip(boxes, scores, pred_classes):
            x1, y1, x2, y2 = box
            im_pred = cv2.rectangle(im_pred, (int(x1), int(y1)), (int(x2), int(y2)), colors["pred"], 2)
            im_pred = cv2.putText(im_pred, f"{class_names[class_id]} ({score:.2f})", 
                                  (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors["pred"], 2)
            if score > max_score:
                max_score = score
                pred_class = class_names[class_id]
        
        if pred_class is None:
            pred_class = "None"
        cv2.putText(im_pred, f"Prediction: {pred_class} (Conf: {max_score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors["pred"], 2)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(cv2.cvtColor(im_gt, cv2.COLOR_BGR2RGB))
        ax1.set_title("Ground Truth", color='green', fontsize=16)
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(im_pred, cv2.COLOR_BGR2RGB))
        ax2.set_title("Prediction", color='yellow', fontsize=16)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"output_images/comparison_{i}.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved image {i+1}/{num_images}")

def main():
    weights_file = "output/training_increased_iter/model_final.pth"
    test_json = "output/test_annotations.json"
    image_dir = "../data/images"

    if not os.path.exists(weights_file):
        print(f"Error: Weights file '{weights_file}' not found.")
        print("Available files in output/training_increased_iter/:")
        for file in os.listdir("output/training_increased_iter/"):
            print(file)
        return

    cfg = setup_cfg(weights_file)
    predictor = DefaultPredictor(cfg)

    DatasetCatalog.register("cesm_test", lambda: load_coco_json(test_json, image_dir))
    MetadataCatalog.get("cesm_test").set(thing_classes=["Benign", "Malignant", "Normal"])

    dataset_dicts = load_coco_json(test_json, image_dir)
    visualize_predictions(predictor, dataset_dicts)

    print(f"Current confidence threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")

    # Evaluate the model
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    evaluator = COCOEvaluator("cesm_test", cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "cesm_test")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)

if __name__ == "__main__":
    main()