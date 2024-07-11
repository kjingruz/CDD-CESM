import torch
import os
import json
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode

def setup_cfg(weights_file):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
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
    evaluator = COCOEvaluator(dataset_name, output_dir="./output_improved")
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(model, val_loader, evaluator)
    
    coco_results = []
    if "bbox" in results:
        for image_id, predictions in results["bbox"].items():
            if isinstance(predictions, dict):
                # Single prediction per image
                coco_results.append({
                    "image_id": int(image_id),
                    "category_id": int(predictions.get("category_id", 0)),
                    "bbox": predictions.get("bbox", []),
                    "score": float(predictions.get("score", 0))
                })
            elif isinstance(predictions, list):
                # Multiple predictions per image
                for pred in predictions:
                    coco_results.append({
                        "image_id": int(image_id),
                        "category_id": int(pred.get("category_id", 0)),
                        "bbox": pred.get("bbox", []),
                        "score": float(pred.get("score", 0))
                    })
            else:
                print(f"Unexpected prediction format for image {image_id}: {type(predictions)}")
    
    with open("output_improved/predictions_coco.json", "w") as f:
        json.dump(coco_results, f)

    print("Predictions saved in COCO format to output_improved/predictions_coco.json")
    return coco_results

def visualize_predictions(predictor, dataset_dicts, num_images=50):
    metadata = MetadataCatalog.get("cesm_test")
    class_names = metadata.thing_classes
    colors = {"gt": (0, 255, 0), "pred": (255, 255, 0)}  # Green for GT, Bright Yellow for predictions

    output_dir = os.path.join("output_improved", "output_images")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, d in enumerate(random.sample(dataset_dicts, num_images)):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        
        im_gt = im.copy()
        im_pred = im.copy()
        
        gt_class = None
        for ann in d["annotations"]:
            box = ann["bbox"]
            im_gt = cv2.rectangle(im_gt, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), colors["gt"], 2)
            class_id = ann["category_id"]
            gt_class = class_names[class_id]
        
        cv2.putText(im_gt, f"Ground Truth: {gt_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors["gt"], 2)
        
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
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(cv2.cvtColor(im_gt, cv2.COLOR_BGR2RGB))
        ax1.set_title("Ground Truth", color='green', fontsize=16)
        ax1.axis('off')
        
        ax2.imshow(cv2.cvtColor(im_pred, cv2.COLOR_BGR2RGB))
        ax2.set_title("Prediction", color='yellow', fontsize=16)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{i}.jpg"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved image {i+1}/{num_images}")

def main():
    weights_file = "output_improved/model_final.pth"
    test_json = "output/test_annotations.json"
    image_dir = "../data/images"

    if not os.path.exists(weights_file):
        print(f"Error: Weights file '{weights_file}' not found.")
        print("Available files in output_improved/:")
        for file in os.listdir("output_improved"):
            print(file)
        return

    cfg = setup_cfg(weights_file)
    predictor = DefaultPredictor(cfg)

    register_coco_instances("cesm_test", {}, test_json, image_dir)
    MetadataCatalog.get("cesm_test").set(thing_classes=["Benign", "Malignant", "Normal"])

    dataset_dicts = load_coco_json(test_json, image_dir)
    
    # Generate predictions in COCO format
    coco_results = generate_predictions_coco(cfg, predictor.model, "cesm_test")

    # Debug print
    print(f"Number of predictions: {len(coco_results)}")
    if coco_results:
        print("Sample prediction:")
        print(json.dumps(coco_results[0], indent=2))

    visualize_predictions(predictor, dataset_dicts)

    print(f"Current confidence threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")

    # Evaluate the model
    evaluator = COCOEvaluator("cesm_test", cfg, False, output_dir="./output_improved/")
    val_loader = build_detection_test_loader(cfg, "cesm_test")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(results)

if __name__ == "__main__":
    main()

