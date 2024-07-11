import torch
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

def generate_predictions_coco(cfg, model, dataset_name):
    evaluator = COCOEvaluator(dataset_name, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(model, val_loader, evaluator)
    
    # Convert results to COCO format
    coco_results = []
    for image_id, predictions in results.items():
        for prediction in predictions:
            coco_results.append({
                "image_id": image_id,
                "category_id": prediction["category_id"],
                "bbox": prediction["bbox"],
                "score": prediction["score"]
            })
    
    # Save predictions to a JSON file
    with open("predictions_coco.json", "w") as f:
        json.dump(coco_results, f)

    print("Predictions saved in COCO format to predictions_coco.json")

def setup_detectron2(train_json, val_json, train_dir, val_dir, output_dir):
    register_coco_instances("cesm_train", {}, train_json, train_dir)
    register_coco_instances("cesm_val", {}, val_json, val_dir)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)
    cfg.DATALOADER.NUM_WORKERS = 2  # Increased from 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2  # Increased from 1
    cfg.SOLVER.BASE_LR = 0.001  # Increased learning rate
    cfg.SOLVER.MAX_ITER = 10000  # Increased from 1000
    cfg.SOLVER.STEPS = (7000, 9000)  # Learning rate decay
    cfg.SOLVER.GAMMA = 0.1  # Learning rate decay factor
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Increased from 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.EVAL_PERIOD = 1000  # Evaluate every 1000 iterations
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

def train_model(cfg):
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
    print("Training completed or stopped.")

def main():
    train_json = './output/train_annotations.json'
    val_json = './output/val_annotations.json'
    image_dir = '../data/images'
    output_dir = './output/training_increased_iter'

    cfg = setup_detectron2(train_json, val_json, image_dir, image_dir, output_dir)
    train_model(cfg)
    # After training the model
    generate_predictions_coco(cfg, trainer.model, "cesm_test")

if __name__ == "__main__":
    main()