import os
import pandas as pd
import numpy as np
import json
import cv2
from sklearn.model_selection import GroupShuffleSplit
import torch
from collections import defaultdict
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, hooks
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from tqdm import tqdm
import requests
import traceback
import sys
import signal
import time
import psutil


def count_excel_classifications(excel_file):
    df = pd.read_excel(excel_file)
    
    print("Excel file shape:", df.shape)
    print("Excel file columns:", df.columns.tolist())
    
    if 'Pathology Classification/ Follow up' not in df.columns:
        raise ValueError("'Pathology Classification/ Follow up' column not found in Excel file")
    
    classification_counts = df['Pathology Classification/ Follow up'].value_counts()
    print("\nClassification counts in Excel file:")
    print(classification_counts)
    
    return classification_counts

def clean_filenames(directory):
    for filename in os.listdir(directory):
        if ' .jpg' in filename:
            new_filename = filename.replace(' .jpg', '.jpg')
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print(f"Renamed: {filename} to {new_filename}")

def load_and_process_data(csv_file, excel_file):
    # Load CSV file (for bbox and segmentation)
    csv_df = pd.read_csv(csv_file)
    csv_df['#filename'] = csv_df['#filename'].apply(lambda x: x.replace(' .jpg', '.jpg').replace('.jpg', ''))
    
    # Group annotations by filename
    csv_dict = defaultdict(list)
    for _, row in csv_df.iterrows():
        csv_dict[row['#filename']].append({
            'region_shape_attributes': row['region_shape_attributes'],
            'region_attributes': row['region_attributes']
        })

    # Load Excel file (for classifications and metadata)
    excel_df = pd.read_excel(excel_file)
    excel_df['Image_name'] = excel_df['Image_name'].str.strip()
    
    print("Classification counts in Excel file:")
    print(excel_df['Pathology Classification/ Follow up'].value_counts())

    # Convert category to numerical values
    category_mapping = {'Benign': 0, 'Malignant': 1, 'Normal': 2}
    excel_df['category_id'] = excel_df['Pathology Classification/ Follow up'].map(category_mapping)

    # Create final dataframe
    final_df = excel_df.copy()
    final_df['has_annotation'] = final_df['Image_name'].isin(csv_dict)
    
    # Add annotation information where available
    for idx, row in final_df.iterrows():
        if row['has_annotation']:
            final_df.at[idx, 'annotations'] = csv_dict[row['Image_name']]
        else:
            final_df.at[idx, 'annotations'] = []

    print("\nFinal dataframe shape:", final_df.shape)
    print("Classification counts in final dataframe:")
    print(final_df['Pathology Classification/ Follow up'].value_counts())
    print("\nNumerical category counts:")
    print(final_df['category_id'].value_counts().sort_index())
    print("\nImages with annotations:", final_df['has_annotation'].sum())
    print("Images without annotations:", (~final_df['has_annotation']).sum())

    return final_df

def create_coco_annotations(df, image_dir):
    images = []
    annotations = []
    categories = [{"id": i, "name": name} for i, name in enumerate(['Benign', 'Malignant', 'Normal'])]
    
    annotation_id = 1
    image_id = 0
    category_counts = {0: 0, 1: 0, 2: 0}  # To keep track of image counts per category
    annotation_counts = {0: 0, 1: 0, 2: 0}  # To keep track of annotation counts per category

    for _, row in df.iterrows():
        image_filename = row['Image_name'] + '.jpg'
        image_path = os.path.join(image_dir, image_filename)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image {image_path}")
            continue
        height, width = img.shape[:2]
        
        images.append({
            "id": image_id,
            "file_name": image_filename,
            "height": int(height),
            "width": int(width)
        })
        
        category_id = int(row['category_id'])
        category_counts[category_id] += 1

        if row['has_annotation']:
            for ann in row['annotations']:
                region_attributes = json.loads(ann['region_shape_attributes'])
                if 'all_points_x' in region_attributes and 'all_points_y' in region_attributes:
                    x = region_attributes['all_points_x']
                    y = region_attributes['all_points_y']
                    bbox = [float(min(x)), float(min(y)), float(max(x) - min(x)), float(max(y) - min(y))]
                    segmentation = [list(map(float, np.array([x, y]).T.flatten()))]
                    area = float(cv2.contourArea(np.array(list(zip(x, y)), dtype=np.int32)))
                else:
                    bbox = [0, 0, float(width), float(height)]
                    segmentation = [[0, 0, width, 0, width, height, 0, height]]
                    area = float(width * height)
                
                annotations.append({
                    "id": int(annotation_id),
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation
                })
                annotation_id += 1
                annotation_counts[category_id] += 1
        else:
            # For images without annotations (e.g., normal images), use full image
            annotations.append({
                "id": int(annotation_id),
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [0, 0, float(width), float(height)],
                "area": float(width * height),
                "iscrowd": 0,
                "segmentation": [[0, 0, width, 0, width, height, 0, height]]
            })
            annotation_id += 1
            annotation_counts[category_id] += 1
        
        image_id += 1
    
    print("Classification counts in COCO annotations:")
    for cat_id, cat_name in enumerate(['Benign', 'Malignant', 'Normal']):
        print(f"{cat_name}: {category_counts[cat_id]} images, {annotation_counts[cat_id]} annotations")
    
    print(f"Total: {sum(category_counts.values())} images, {sum(annotation_counts.values())} annotations")

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

def split_dataset(df, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df['Patient_ID']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    train_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, val_idx = next(train_gss.split(train_df, groups=train_df['Patient_ID']))
    
    final_train_df = train_df.iloc[train_idx]
    val_df = train_df.iloc[val_idx]
    
    print("\nClassification counts after splitting:")
    for split_name, split_df in [('Train', final_train_df), ('Validation', val_df), ('Test', test_df)]:
        print(f"{split_name} set:")
        print(split_df['category_id'].value_counts().sort_index())
        print(f"Total images: {len(split_df)}")
        print()

    return final_train_df, val_df, test_df

def save_coco_json(annotations, file_path):
    with open(file_path, 'w') as f:
        json.dump(annotations, f, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def download_with_progress(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            size = file.write(data)
            progress_bar.update(size)
    
    progress_bar.close()

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=300):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    finally:
        signal.alarm(0)
    return result

def setup_detectron2(train_json, val_json, train_dir, val_dir, output_dir):
    register_coco_instances("cesm_train", {}, train_json, train_dir)
    register_coco_instances("cesm_val", {}, val_json, val_dir)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)
    cfg.DATALOADER.NUM_WORKERS = 0
    
    # Use model_zoo.get_checkpoint_url for direct loading
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001  # Lowered learning rate
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = []  # Remove learning rate decay
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Benign, Malignant, Normal
    cfg.TEST.EVAL_PERIOD = 1000

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative_range"
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    cfg.INPUT.RANDOM_FLIP = "horizontal"

    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.GAMMA = 0.1

    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def train_model(cfg):
    setup_logger()
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    print("Starting training...")
    trainer.train()
    print("Training completed.")

    if trainer and hasattr(trainer, 'model'):
        print("Model architecture:")
        print(trainer.model)

        print("\nModel parameters:")
        total_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"Total parameters: {total_params}")

        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

    print("Memory usage:")
    process = psutil.Process()
    print(f"Memory used: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def main():
    csv_file = '../data/Radiology_hand_drawn_segmentations_v2.csv'
    excel_file = '../data/Radiology-manual-annotations.xlsx'
    image_dir = '../data/images'
    output_dir = './output'

    # Clean filenames in the image directory
    clean_filenames(image_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    df = load_and_process_data(csv_file, excel_file)

    # Split dataset
    train_df, val_df, test_df = split_dataset(df)

    # Create and save COCO annotations for each split
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        coco_annotations = create_coco_annotations(split_df, image_dir)
        json_path = os.path.join(output_dir, f'{split_name}_annotations.json')
        save_coco_json(coco_annotations, json_path)
        print(f"Saved {split_name} annotations to {json_path}")

        csv_path = os.path.join(output_dir, f'{split_name}_annotations.csv')
        split_df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} CSV to {csv_path}")

    # Setup Detectron2
    train_json = os.path.join(output_dir, 'train_annotations.json')
    val_json = os.path.join(output_dir, 'val_annotations.json')
    cfg = setup_detectron2(train_json, val_json, image_dir, image_dir, output_dir)

    # Train the model
    print("Preparing to train model...")
    train_model(cfg)

    print("All processes completed.")

if __name__ == "__main__":
    main()