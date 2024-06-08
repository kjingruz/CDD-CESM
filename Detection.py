import os
import shutil
import pandas as pd
import csv
import json
import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
import torch
import copy
import imgaug.augmenters as iaa
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetMapper, build_detection_train_loader

torch.multiprocessing.set_sharing_strategy('file_system')
setup_logger()

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    _, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(otsu_thresh)
        cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        x, y, w, h = cv2.boundingRect(max_contour)
        cropped_image = masked_image[y:y+h, x:x+w]
    else:
        cropped_image = image

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    equalized_image = clahe.apply(cropped_image)
    
    return equalized_image

def preprocess_and_save(image_path, save_path):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        cv2.imwrite(save_path, preprocessed_image)
    else:
        print(f"Skipping image: {image_path} due to preprocessing failure")

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
        
        # Remove the path from the image name
        image_name = os.path.basename(image_name)
        
        if classification == 'Benign':
            classifications[image_name] = 0  # Use integer labels
        elif classification == 'Malignant':
            classifications[image_name] = 1
        else:
            classifications[image_name] = 2
        
        print(f"Classify: {image_name} as {classification} ({classifications[image_name]})")
    
    return classifications

classifications = classify_images(df_annotations)

os.makedirs('./data/annotated_images', exist_ok=True)

images_info = []
annotations_info = []
category_id_mapping = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

# Track all classifications processed
tracked_category_counts = {'Benign': 0, 'Malignant': 0, 'Normal': 0}

# Create a dictionary from the Excel sheet for quick lookup
excel_classifications = df_annotations.set_index('Image_name')['Pathology Classification/ Follow up'].to_dict()

for filename, points in annotations_by_filename.items():
    image_path = f'./data/images/{filename}'
    if os.path.exists(image_path):
        # Use the filename (without extension) to look up the classification
        classification = classifications.get(os.path.basename(os.path.splitext(filename)[0]), 2)  # Default to 'Normal'
        annotated_image_path = f'./data/annotated_images/{filename}'
        preprocess_and_save(image_path, annotated_image_path)
        image = cv2.imread(annotated_image_path)
        if image is None:
            print(f"Failed to load annotated image: {annotated_image_path}")
            continue
        height, width = image.shape[:2]
        if points:
            for annotation in points:
                all_points_x = annotation['all_points_x']
                all_points_y = annotation['all_points_y']
                if all_points_x and all_points_y:  # Ensure the points are not empty
                    segmentation = [list(np.array([all_points_x, all_points_y]).T.flatten())]
                    bbox = [int(min(all_points_x)), int(min(all_points_y)), int(max(all_points_x) - min(all_points_x)), int(max(all_points_y) - min(all_points_y))]  # Convert to int
                    area = int(cv2.contourArea(np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)))  # Convert to int
                    
                    if area > 0:
                        annotations_info.append({
                            "id": len(annotations_info) + 1,
                            "image_id": len(images_info) + 1,
                            "category_id": classification,  # Use classification instead of category_id
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0
                        })
        else:
            # Create a dummy annotation for Normal images without actual annotations
            annotations_info.append({
                "id": len(annotations_info) + 1,
                "image_id": len(images_info) + 1,
                "category_id": classification,  # Normal
                "segmentation": [[]],  # Empty segmentation
                "area": 0,
                "bbox": [0, 0, 0, 0],  # Dummy bbox
                "iscrowd": 0
            })
        images_info.append({
            "id": len(images_info) + 1,
            "file_name": filename,
            "width": int(width),
            "height": int(height)
        })
        tracked_category_counts[category_id_mapping[classification]] += 1
        print(f"Processed image: {filename} with classification: {category_id_mapping[classification]}")

categories_info = [{"id": 0, "name": "Benign"}, {"id": 1, "name": "Malignant"}, {"id": 2, "name": "Normal"}]

# Verify classifications
for image_info in images_info:
    image_name = os.path.basename(os.path.splitext(image_info['file_name'])[0])
    assigned_classification = category_id_mapping[classifications.get(image_name, 2)]
    excel_classification = excel_classifications.get(image_name, 'Normal')
    if assigned_classification != excel_classification:
        print(f"Discrepancy found: {image_name} assigned as {assigned_classification}, but Excel sheet has {excel_classification}")

all_images = [f for f in os.listdir('./data/images') if f.endswith(('.jpg', '.jpeg'))]
for image_file in all_images:
    if image_file not in annotated_images:
        image_path = f'./data/images/{image_file}'
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            height, width = image.shape[:2]
            images_info.append({
                "id": len(images_info) + 1,
                "file_name": image_file,
                "width": int(width),
                "height": int(height)
            })
            # Create a dummy annotation for Normal images without actual annotations
            annotations_info.append({
                "id": len(annotations_info) + 1,
                "image_id": len(images_info),
                "category_id": 2,  # Normal
                "segmentation": [[]],  # Empty segmentation
                "area": 0,
                "bbox": [0, 0, 0, 0],  # Dummy bbox
                "iscrowd": 0
            })
            print(f"Added unannotated image: {image_file}")
            tracked_category_counts['Normal'] += 1  # Update the counts for normal images

print("Tracked category counts:", tracked_category_counts)

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
    elif isinstance(data, (np.generic, np.ndarray)):
        return data.tolist() if isinstance(data, np.ndarray) else data.item()
    else:
        return data

coco_annotation = convert_to_native_types(coco_annotation)
annotations_info = convert_to_native_types(annotations_info)

with open('./data/annotations.json', 'w') as f:
    json.dump(coco_annotation, f)

with open('./data/classifications.json', 'w') as f:
    json.dump(classifications, f)

with open('./data/annotations_info.json', 'w') as f:
    json.dump(annotations_info, f)

patient_ids = list(set([img['file_name'].split("_")[0] for img in images_info]))
group_split = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
train_idx, test_idx = next(group_split.split(patient_ids, groups=patient_ids))

train_patient_ids = [patient_ids[idx] for idx in train_idx]
test_patient_ids = [patient_ids[idx] for idx in test_idx]

patient_groups = {}
for img in images_info:
    patient_id = img['file_name'].split("_")[0]
    if patient_id not in patient_groups:
        patient_groups[patient_id] = []
    patient_groups[patient_id].append(img)

train_images = [image for patient_id in train_patient_ids for image in patient_groups[patient_id]]
test_images = [image for patient_id in test_patient_ids for image in patient_groups[patient_id]]

train_counts = {'Benign': 0, 'Malignant': 0, 'Normal': 0}
test_counts = {'Benign': 0, 'Malignant': 0, 'Normal': 0}

for image in train_images:
    image_name = os.path.basename(os.path.splitext(image['file_name'])[0])
    classification = classifications.get(image_name, 2)  # Default to 'Normal'
    train_counts[category_id_mapping[classification]] += 1

for image in test_images:
    image_name = os.path.basename(os.path.splitext(image['file_name'])[0])
    classification = classifications.get(image_name, 2)  # Default to 'Normal'
    test_counts[category_id_mapping[classification]] += 1

print("Train counts:", train_counts)
print("Test counts:", test_counts)

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
    print(f"Created COCO json: {dest_file}")

os.makedirs('./data/train', exist_ok=True)
os.makedirs('./data/valid', exist_ok=True)

create_coco_json(train_images, annotations_info, categories_info, './data/train/_annotations.coco.json')
create_coco_json(test_images, annotations_info, categories_info, './data/valid/_annotations.coco.json')

def move_files(images, src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    for img in images:
        src = os.path.join(src_folder, img['file_name'])
        dest = os.path.join(dest_folder, img['file_name'])
        if os.path.exists(src):
            shutil.copyfile(src, dest)
            print(f"Copied {src} to {dest}")
        else:
            print(f"File {src} does not exist")

move_files(train_images, './data/images', './data/train')
move_files(test_images, './data/images', './data/valid')

register_coco_instances("my_dataset_train", {}, "./data/train/_annotations.coco.json", "./data/train")
register_coco_instances("my_dataset_val", {}, "./data/valid/_annotations.coco.json", "./data/valid")

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

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
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

output_dir = "/Users/kjingruz/Library/CloudStorage/OneDrive-McMasterUniversity/Research/Saha/Second Batch"
os.makedirs(output_dir, exist_ok=True)
cfg.OUTPUT_DIR = output_dir

trainer = TrainerWithCustomLoader(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
