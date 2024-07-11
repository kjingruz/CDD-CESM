import argparse
from coco_froc_analysis.froc import generate_froc_curve, generate_bootstrap_froc_curves

def main():
    parser = argparse.ArgumentParser(description='Run COCO FROC analysis')
    parser.add_argument('--gt_ann', required=True, help='Path to ground truth annotations')
    parser.add_argument('--pr_ann', required=True, help='Path to prediction annotations')
    parser.add_argument('--bootstrap', type=int, default=0, help='Number of bootstrap samples (0 for single run)')
    parser.add_argument('--use_iou', action='store_true', help='Use IoU for matching')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IoU threshold')
    parser.add_argument('--n_sample_points', type=int, default=100, help='Number of points to sample for FROC curve')
    parser.add_argument('--plot_title', default='FROC Analysis', help='Title for the plot')
    parser.add_argument('--plot_output_path', default='froc_analysis.png', help='Output path for the plot')
    
    args = parser.parse_args()

    if args.bootstrap > 0:
        generate_bootstrap_froc_curves(
            gt_ann=args.gt_ann,
            pr_ann=args.pr_ann,
            n_bootstrap_samples=args.bootstrap,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title=args.plot_title,
            plot_output_path=args.plot_output_path
        )
    else:
        generate_froc_curve(
            gt_ann=args.gt_ann,
            pr_ann=args.pr_ann,
            use_iou=args.use_iou,
            iou_thres=args.iou_thres,
            n_sample_points=args.n_sample_points,
            plot_title=args.plot_title,
            plot_output_path=args.plot_output_path
        )

    print(f"FROC analysis complete. Plot saved to {args.plot_output_path}")

if __name__ == "__main__":
    main()