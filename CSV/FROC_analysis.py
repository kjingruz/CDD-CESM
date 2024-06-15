from coco_froc_analysis.froc import generate_froc_curve

def run_froc_analysis():
    gt_ann = 'path/to/ground_truth.json'
    pr_ann = 'path/to/predictions.json'

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
