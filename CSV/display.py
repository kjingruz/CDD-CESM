import os
import cv2
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import torch

def load_and_predict(image_paths, predictor, metadata):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_path}")
            continue
        
        outputs = predictor(image)
        
        # Print outputs for debugging
        print(f"Predictions for {image_path}: {outputs}")
        
        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        plt.figure(figsize=(12, 12))
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.title(os.path.basename(image_path))
        plt.axis('off')
        plt.show()  # Ensure the plot is shown immediately

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join("../output/train_lowtime", "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Adjust based on your dataset
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_test")
    
    # Load some test images
    test_image_dir = "../../data/test"
    test_images = []
    for subfolder in ["normal", "malignant", "benign"]:
        subfolder_path = os.path.join(test_image_dir, subfolder)
        test_images += [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.endswith(".jpg")]
    
    # Display predictions for a few test images
    load_and_predict(test_images[:5], predictor, test_metadata)

if __name__ == "__main__":
    main()
