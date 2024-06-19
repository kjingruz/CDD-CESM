import os
import random
import cv2
import torch
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from sklearn.metrics import f1_score, accuracy_score
from config_and_train import setup_cfg

# Define the function to get class name from class id
def get_class_name_from_id(class_id):
    class_names = {0: "Benign", 1: "Malignant", 2: "Normal"}
    return class_names.get(class_id, "Unknown")

# Visualize the results where ground truth and prediction agree
def visualize_predictions(predictor, dataset_name, num_images=20):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    for data in random.sample(dataset_dicts, num_images):
        im = cv2.imread(data["file_name"])
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")

        # Filter out low confidence predictions
        high_conf_instances = instances[instances.scores >= 0.5]

        if len(high_conf_instances) > 0 and data['annotations']:
            predicted_class_id = high_conf_instances.pred_classes[0].item()
            ground_truth_class_id = data['annotations'][0]['category_id'] - 1  # Adjust for zero-indexed class ID
            if predicted_class_id == ground_truth_class_id:
                v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8)
                out_pred = v.draw_instance_predictions(high_conf_instances)
                out_gt = v.draw_dataset_dict(data)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                ax1.imshow(out_gt.get_image()[:, :, ::-1])
                ax1.set_title(f"Ground Truth: {get_class_name_from_id(ground_truth_class_id)}")
                ax1.axis('off')

                ax2.imshow(out_pred.get_image()[:, :, ::-1])
                ax2.set_title(f"Predicted: {get_class_name_from_id(predicted_class_id)}")
                ax2.axis('off')

                plt.show()

# Save predictions to JSON
def save_predictions_to_json(predictor, dataset_name, output_file='predictions.json'):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    predictions = []

    for data in dataset_dicts:
        im = cv2.imread(data["file_name"])
        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        
        for idx, box in enumerate(instances.pred_boxes):
            prediction = {
                "image_id": data["image_id"],
                "category_id": instances.pred_classes[idx].item() + 1,  # Category IDs start from 1 in COCO format
                "bbox": box.tolist(),
                "score": instances.scores[idx].item()
            }
            predictions.append(prediction)

    with open(output_file, 'w') as f:
        json.dump(predictions, f)

    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    cfg = setup_cfg()

    # Perform inference with the trained model
    predictor = DefaultPredictor(cfg)

    # Visualize predictions
    visualize_predictions(predictor, "my_dataset_test")

    # Evaluation with COCO Evaluator
    evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./Last/output/")
    val_loader = build_detection_test_loader(cfg, "my_dataset_test", num_workers=0)  # Disable multiprocessing
    inference_on_dataset(predictor.model, val_loader, evaluator)

    # Save predictions
    save_predictions_to_json(predictor, "my_dataset_test")
