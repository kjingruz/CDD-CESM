from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import os
import json
import random
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

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

def draw_box(img, box, color, thickness=2):
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img

def draw_text(img, text, pos, color, scale=0.5, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    return img

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
            im_gt = draw_box(im_gt, box, colors["gt"])
            class_id = ann["category_id"]
            gt_class = class_names[class_id]
        
        # Draw title for ground truth
        cv2.putText(im_gt, f"Ground Truth: {gt_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colors["gt"], 2)
        
        # Draw predicted boxes and labels
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        max_score = 0
        pred_class = None
        for box, score, class_id in zip(boxes, scores, pred_classes):
            x, y, w, h = box
            im_pred = draw_box(im_pred, [x, y, w - x, h - y], colors["pred"])
            im_pred = draw_text(im_pred, f"{class_names[class_id]} ({score:.2f})", 
                                (int(x), int(y - 10)), colors["pred"], thickness=2)
            if score > max_score:
                max_score = score
                pred_class = class_names[class_id]
        
        # Draw title for prediction
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

if __name__ == "__main__":
    main()