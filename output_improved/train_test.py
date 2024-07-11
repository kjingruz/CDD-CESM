import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
import os

def setup_detectron2(train_json, val_json, train_dir, val_dir, output_dir):
    register_coco_instances("cesm_train", {}, train_json, train_dir)
    register_coco_instances("cesm_val", {}, val_json, val_dir)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.EVAL_PERIOD = 200
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = "cpu"
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
    output_dir = './output/training_test'

    cfg = setup_detectron2(train_json, val_json, image_dir, image_dir, output_dir)
    train_model(cfg)

if __name__ == "__main__":
    main()