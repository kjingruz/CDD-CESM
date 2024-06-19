import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import BoxMode
import torch
from coco_froc_analysis.froc import generate_froc_curve

def load_coco_json(json_file, image_root):
    with open(json_file) as f:
        coco_dict = json.load(f)
    
    dataset_dicts = []
    for img in coco_dict['images']:
        record = {}
        record['file_name'] = os.path.join(image_root, img['file_name'])
        record['image_id'] = img['id']
        record['height'] = img['height']
        record['width'] = img['width']
        record['annotations'] = []
        
        for ann in coco_dict['annotations']:
            if ann['image_id'] == img['id']:
                obj = {
                    'bbox': ann['bbox'],
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'segmentation': ann.get('segmentation', []),
                    'category_id': ann['category_id'] - 1,  # Adjust category_id to start from 0
                    'iscrowd': ann.get('iscrowd', 0),
                    'area': ann['area']
                }
                record['annotations'].append(obj)
        
        dataset_dicts.append(record)
    
    return dataset_dicts

def register_dataset(name, json_file, image_dir):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_dir))
    MetadataCatalog.get(name).set(thing_classes=["Benign", "Malignant", "Normal"])

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join("../output/train_lowtime/2nd_Attempt", "model_final.pth")
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for this model
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Adjust the number of classes to 3
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "my_dataset_test")

    # Save predictions in COCO JSON format
    evaluation_results = inference_on_dataset(predictor.model, test_loader, evaluator)
    print(evaluation_results)
    
    # Save the predictions to a JSON file
    with open('predictions.json', 'w') as f:
        json.dump(evaluator._predictions, f, indent=4)

    # Run FROC analysis
    run_froc_analysis()

def run_froc_analysis():
    gt_ann = '/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/data/test_annotations.json'  # Path to your ground truth annotations
    pr_ann = 'predictions.json'  # Path to your generated predictions

    generate_froc_curve(
        gt_ann=gt_ann,
        pr_ann=pr_ann,
        use_iou=True,
        iou_thres=0.5,
        n_sample_points=100,
        plot_title='FROC Curve',
        plot_output_path='froc_curve.png'
    )

if __name__ == "__main__":
    annotations_test_json = os.path.abspath("/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/data/test_annotations.json")
    test_image_dir = os.path.abspath("/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/data/test")
    print(f"Annotations JSON path: {annotations_test_json}")
    print(f"Test image directory path: {test_image_dir}")
    register_dataset("my_dataset_test", annotations_test_json, test_image_dir)
    main()
