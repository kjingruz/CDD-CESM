import os
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

# Define the configuration setup
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.DATALOADER.NUM_WORKERS = 0  # Disable multiprocessing to avoid shared memory issue
    cfg.MODEL.WEIGHTS = os.path.join("/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/Last/output", "model_final.pth")  # Load the trained weights
    cfg.SOLVER.IMS_PER_BATCH = 4  # Increase batch size
    cfg.SOLVER.BASE_LR = 0.0025  # Increase learning rate
    cfg.SOLVER.MAX_ITER = 8000  # Increase max iterations for potentially better training
    cfg.SOLVER.STEPS = []  # Do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Increase batch size per image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Update this based on your classes (Benign, Malignant, Normal)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.TEST.EVAL_PERIOD = 1000  # Evaluate less frequently to save time

    # Set up data augmentation
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # Set the output directory
    cfg.OUTPUT_DIR = "/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/Last/output"

    # Register the dataset
    register_coco_instances("my_dataset_test", {}, "/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/data/test_annotations.json", "/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/data/test")

    return cfg

if __name__ == "__main__":
    setup_cfg()
    print("Configuration and dataset registration complete.")
