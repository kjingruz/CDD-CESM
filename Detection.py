# Import necessary libraries
import os
import shutil
import pandas as pd
import csv
import json
import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
import torch
import copy
import imgaug.augmenters as iaa
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper, build_detection_train_loader

# Set the sharing strategy to 'file_system'
torch.multiprocessing.set_sharing_strategy('file_system')

# Load and prepare annotations and classifications (this part is unchanged)
segmentation_file = './data/Radiology_hand_drawn_segmentations_v2.csv'
annotations = []
with open(segmentation_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['region_shape_attributes']:  # Only process rows with region_shape_attributes
            region_shape_attributes = json.loads(row['region_shape_attributes'])
            annotations.append({
                'filename': row['#filename'],
                'points': {
                    'all_points_x': region_shape_attributes.get('all_points_x', []),
                    'all_points_y': region_shape_attributes.get('all_points_y', [])
                }
            })

annotations_by_filename = {}
for annotation in annotations:
    if annotation['filename'] not in annotations_by_filename:
        annotations_by_filename[annotation['filename']] = []
    annotations_by_filename[annotation['filename']].append(annotation['points'])

annotated_images = list(annotations_by_filename.keys())

annotations_file = './data/Radiology-manual-annotations.xlsx'
df_annotations = pd.read_excel(annotations_file)

def classify_images(df):
    classifications = {}
    for index, row in df.iterrows():
        image_name = row['Image_name']
        classification = row['Pathology Classification/ Follow up']
        
        if classification == 'Benign':
            classifications[image_name] = 0  # Use integer labels
        elif classification == 'Malignant':
            classifications[image_name] = 1
        else:
            classifications[image_name] = 2
    
    return classifications

classifications = classify_images(df_annotations)

os.makedirs('./data/annotated_images', exist_ok=True)

images_info = []
annotations_info = []
category_id_mapping = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

for filename, points in annotations_by_filename.items():
    image_path = f'./data/images/{filename}'
    if os.path.exists(image_path):
        classification = classifications.get(os.path.splitext(filename)[0], 2)  # Default to 'Normal'
        annotated_image_path = f'./data/annotated_images/{filename}'
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        for annotation in points:
            all_points_x = annotation['all_points_x']
            all_points_y = annotation['all_points_y']
            if all_points_x and all_points_y:  # Ensure the points are not empty
                segmentation = [list(np.array([all_points_x, all_points_y]).T.flatten())]
                bbox = [int(min(all_points_x)), int(min(all_points_y)), int(max(all_points_x) - min(all_points_x)), int(max(all_points_y) - min(all_points_y))]  # Convert to int
                area = int(cv2.contourArea(np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)))  # Convert to int
                
                annotations_info.append({
                    "id": len(annotations_info) + 1,
                    "image_id": len(images_info) + 1,
                    "category_id": classification,  # Use classification instead of category_id
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
        images_info.append({
            "id": len(images_info) + 1,
            "file_name": filename,
            "width": int(width),
            "height": int(height)
        })
        cv2.imwrite(annotated_image_path, image)

categories_info = [{"id": 0, "name": "Benign"}, {"id": 1, "name": "Malignant"}, {"id": 2, "name": "Normal"}]

all_images = [f for f in os.listdir('./data/images') if f.endswith(('.jpg', '.jpeg'))]
for image_file in all_images:
    if image_file not in annotated_images:
        image_path = f'./data/images/{image_file}'
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            height, width = image.shape[:2]
            images_info.append({
                "id": len(images_info) + 1,
                "file_name": image_file,
                "width": int(width),
                "height": int(height)
            })

coco_annotation = {
    "images": images_info,
    "annotations": annotations_info,
    "categories": categories_info
}

def convert_to_native_types(data):
    if isinstance(data, list):
        return [convert_to_native_types(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_native_types(value) for key, value in data.items()}
    elif isinstance(data, np.generic):
        return data.item()
    else:
        return data

coco_annotation = convert_to_native_types(coco_annotation)

with open('./data/annotated_images/annotations.json', 'w') as f:
    json.dump(coco_annotation, f)

patient_groups = {}
for image_info in images_info:
    patient_id = image_info['file_name'].split('_')[0]
    if patient_id not in patient_groups:
        patient_groups[patient_id] = []
    patient_groups[patient_id].append(image_info)

group_kfold = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
patient_ids = list(patient_groups.keys())
train_idx, test_idx = next(group_kfold.split(X=patient_ids, groups=patient_ids))

train_patient_ids = [patient_ids[idx] for idx in train_idx]
test_patient_ids = [patient_ids[idx] for idx in test_idx]

train_images = [img for pid in train_patient_ids for img in patient_groups[pid]]
test_images = [img for pid in test_patient_ids for img in patient_groups[pid]]

def create_coco_json(images, annotations, categories, dest_file):
    image_ids = [img['id'] for img in images]
    filtered_annotations = [anno for anno in annotations if anno['image_id'] in image_ids]
    coco_data = {
        "images": images,
        "annotations": filtered_annotations,
        "categories": categories
    }
    coco_data = convert_to_native_types(coco_data)
    with open(dest_file, 'w') as f:
        json.dump(coco_data, f)

os.makedirs('./data/train', exist_ok=True)
os.makedirs('./data/valid', exist_ok=True)

create_coco_json(train_images, annotations_info, categories_info, './data/train/_annotations.coco.json')
create_coco_json(test_images, annotations_info, categories_info, './data/valid/_annotations.coco.json')

def move_files(images, src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for img in images:
        src = os.path.join(src_folder, img['file_name'])
        dest = os.path.join(dest_folder, img['file_name'])
        shutil.copyfile(src, dest)

move_files(train_images, './data/images', './data/train')
move_files(test_images, './data/images', './data/valid')

train_counts = {'Benign': 0, 'Malignant': 0, 'Normal': 0}
test_counts = {'Benign': 0, 'Malignant': 0, 'Normal': 0}

for image in train_images:
    image_name = os.path.splitext(image['file_name'])[0]
    classification = classifications.get(image_name, 2)  # Default to 'Normal'
    train_counts[category_id_mapping[classification]] += 1

for image in test_images:
    image_name = os.path.splitext(image['file_name'])[0]
    classification = classifications.get(image_name, 2)  # Default to 'Normal'
    test_counts[category_id_mapping[classification]] += 1

print("Train counts:", train_counts)
print("Test counts:", test_counts)

register_coco_instances("my_dataset_train", {}, "./data/train/_annotations.coco.json", "./data/train")
register_coco_instances("my_dataset_val", {}, "./data/valid/_annotations.coco.json", "./data/valid")

setup_logger()

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

# Configuration setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 4  # Increase for better performance
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 8  # Increase batch size, assuming enough GPU memory
cfg.SOLVER.BASE_LR = 0.01  # Increase learning rate for potentially faster convergence
cfg.SOLVER.MAX_ITER = 5000  # Increase max iterations for potentially better training
cfg.SOLVER.STEPS = []  # Do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # Increase batch size per image, assuming enough GPU memory
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Update this based on your classes (Benign, Malignant, Normal)
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.TEST.EVAL_PERIOD = 1000  # Evaluate less frequently to save time

# Define the output directory
output_dir = "/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/computecanada/output"
os.makedirs(output_dir, exist_ok=True)
cfg.OUTPUT_DIR = output_dir

# Initialize trainer and start training
trainer = TrainerWithCustomLoader(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Save the trained model weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

