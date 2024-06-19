import os
from coco_froc_analysis.froc import generate_froc_curve

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
    run_froc_analysis()
