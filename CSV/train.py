import os
from detectron2.engine import DefaultTrainer, hooks
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
import imgaug.augmenters as iaa
import torch
import pandas as pd
import json
import numpy as np
from detectron2.data import DatasetMapper
import copy
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

torch.multiprocessing.set_sharing_strategy('file_system')

def load_dataset(image_dir):
    categories = ["benign", "malignant", "normal"]
    category_id_mapping = { "benign": 0, "malignant": 1, "normal": 2 }
    
    dataset_dicts = []
    for category in categories:
        category_path = os.path.join(image_dir, category)
        if not os.path.exists(category_path):
            print(f"Directory {category_path} does not exist, skipping.")
            continue
        
        for filename in os.listdir(category_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            record = {}
            filepath = os.path.join(category_path, filename)
            height, width = utils.read_image(filepath).shape[:2]
            
            record["file_name"] = filepath
            record["image_id"] = len(dataset_dicts)
            record["height"] = height
            record["width"] = width
            record["annotations"] = [{
                "bbox": [0, 0, width, height],  # No specific bbox provided, default to the whole image
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": [],  # No segmentation provided
                "category_id": category_id_mapping[category],
                "iscrowd": 0,
                "area": width * height
            }]
            dataset_dicts.append(record)
    
    return dataset_dicts

def register_dataset(name, image_dir):
    DatasetCatalog.register(name, lambda: load_dataset(image_dir))
    MetadataCatalog.get(name).set(thing_classes=["Benign", "Malignant", "Normal"])

def get_imgaug_transforms():
    return iaa.Sequential([
        iaa.Rotate((-45, 45)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Multiply((0.8, 1.2)),
        iaa.LinearContrast((0.8, 1.2)),
        iaa.Affine(scale=(0.8, 1.2), translate_percent=(-0.2, 0.2), rotate=(-45, 45))
    ])

class ImgaugMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.augmentations = get_imgaug_transforms()
        self.img_format = cfg.INPUT.FORMAT

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        aug_output = self.augmentations(image=image)
        image = aug_output.astype("float32")

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1))

        annos = [utils.transform_instance_annotations(annotation, [], image.shape[:2])
                 for annotation in dataset_dict.pop("annotations")]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

class TrainerWithCustomLoader(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=ImgaugMapper(cfg, is_train=True))

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, hooks.EvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            lambda: self.test(self.cfg, self.model, self.build_evaluator(self.cfg, "my_dataset_val"))
        ))
        return hooks

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.DATASETS.VAL = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 0  # Disable multiprocessing to avoid shared memory issue
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 4  # Increase batch size
    cfg.SOLVER.BASE_LR = 0.0025  # Increase learning rate
    cfg.SOLVER.MAX_ITER = 8000  # Increase max iterations for potentially better training
    cfg.SOLVER.STEPS = []  # Do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # Increase batch size per image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Update this based on your classes (Benign, Malignant, Normal)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.TEST.EVAL_PERIOD = 1000  # Evaluate less frequently to save time

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.GAMMA = 0.05

    output_dir = "../output/train"
    os.makedirs(output_dir, exist_ok=True)
    cfg.OUTPUT_DIR = output_dir

    trainer = TrainerWithCustomLoader(cfg)
    trainer.resume_or_load(resume=False)
    try:
        trainer.train()
    except Exception as e:
        print(f"Exception during training: {e}")

if __name__ == "__main__":
    register_dataset("my_dataset_train", "../../data/train")
    register_dataset("my_dataset_val", "../../data/valid")
    register_dataset("my_dataset_test", "../../data/test")
    main()
