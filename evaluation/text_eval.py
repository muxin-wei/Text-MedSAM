import os
import argparse
import pandas as pd
import numpy as np
import cc3d
from collections import OrderedDict
from skimage import segmentation
from scipy.optimize import linear_sum_assignment
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Ensure SurfaceDice is accessible


try:
    from src.dataset.utils import process_input
    from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
except ImportError:
    print("Error: SurfaceDice module not found. Please ensure 'SurfaceDice.py' is in the same directory or PYTHONPATH.")
    import sys
    sys.exit(1)

join = os.path.join

def compute_multi_class_dsc(gt, seg, label_ids):
    # Filter only labels present in both the requested IDs and the GT
    present_labels = set(np.unique(gt)[1:]) & set(label_ids)
    if not present_labels:
        return np.nan
        
    dsc = []
    for i in present_labels:
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.nanmean(dsc)

def compute_multi_class_nsd(gt, seg, spacing, label_ids, tolerance=2.0):
    present_labels = set(np.unique(gt)[1:]) & set(label_ids)
    if not present_labels:
        return np.nan

    nsd = []
    for i in present_labels:
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(gt_i, seg_i, spacing_mm=spacing)
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.nanmean(nsd)

def _label_overlap(x, y):
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(), 1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    matched_pairs = [(t, p) for t, p, ok in zip(true_ind, pred_ind, match_ok) if ok]
    return tp, matched_pairs

def eval_tp_fp_fn(masks_true, masks_pred, threshold=0.5):
    num_inst_gt = np.max(masks_true)
    num_inst_seg = np.max(masks_pred)
    if num_inst_seg > 0:
        iou = _intersection_over_union(masks_true, masks_pred)[1:, 1:]
        tp, matched_pairs = _true_positive(iou, threshold)
        fp = num_inst_seg - tp
        fn = num_inst_gt - tp
    else:
        tp = 0
        fp = 0
        fn = 0
        matched_pairs = None
    return tp, fp, fn, matched_pairs

def main():
    parser = argparse.ArgumentParser(description='Offline Segmentation Evaluation')
    # Path to the Ground Truth (Images usually reside here or nearby, but we need the .npz structure for metadata)
    parser.add_argument('-i', '--test_img_path', default='/root/autodl-tmp/dataset/sample/img', type=str, help='Path to original test images (for spacing/metadata)')
    parser.add_argument('-val_gts', '--validation_gts_path', default='/root/autodl-tmp/dataset/sample/sample_gt', type=str, help='Path to GT .npz files')
    # Path to YOUR pre-computed predictions
    parser.add_argument('-p', '--pred_path', default='/root/autodl-tmp/dataset/sample/seg/epoch=0035-step=042000', type=str, help='Path to your prediction .npz files')
    parser.add_argument('-o', '--save_path', default='./evaluation_results', type=str, help='Path to save CSV results')
    parser.add_argument('--team_name', default='MyModel', type=str, help='Name to use for the output CSV file')
    
    args = parser.parse_args()

    test_img_path = args.test_img_path
    validation_gts_path = args.validation_gts_path
    pred_path = args.pred_path
    save_path = args.save_path
    teamname = args.team_name

    os.makedirs(save_path, exist_ok=True)
    
    # We assume we iterate over the GT folder or Image folder to ensure we cover the test set
    # Using test_img_path to get the list of cases to evaluate
    test_cases = sorted([f for f in os.listdir(test_img_path) if f.endswith('.npz')])

    print(f"Found {len(test_cases)} cases to evaluate.")
    print(f"Predictions directory: {pred_path}")
    print(f"Ground Truth directory: {validation_gts_path}")

    # Initialize metrics
    metric = OrderedDict()
    metric['CaseName'] = []
    # Removed RunningTime since we are doing offline eval
    metric['DSC'] = []
    metric['NSD'] = []
    metric['F1'] = []
    metric['DSC_TP'] = []

    missing_files = []

    for case in test_cases:
        print(f"Evaluating: {case} ...", end=" ")
        
        # Paths
        img_file_path = join(test_img_path, case)
        gt_file_path = join(validation_gts_path, case)
        pred_file_path = join(pred_path, case)

        # Check if prediction exists
        if not os.path.exists(pred_file_path):
            print("Prediction file NOT FOUND")
            missing_files.append(case)
            metric['CaseName'].append(case)
            metric['DSC'].append(np.nan)
            metric['NSD'].append(np.nan)
            metric['F1'].append(np.nan)
            metric['DSC_TP'].append(np.nan)
            continue

        try:
            # 1. Load Data
            # GT (Mask)
            gt_data = np.load(gt_file_path, allow_pickle=True)
            if 'gts' in gt_data:
                gt_npz = gt_data['gts'].astype(np.uint8)
            else:
                # Fallback if keys are different (e.g. if it's just the array)
                gt_npz = gt_data[list(gt_data.keys())[0]].astype(np.uint8)

            # Prediction (Mask)
            pred_data = np.load(pred_file_path, allow_pickle=True)
            # Handle different saving keys in prediction files
            if 'segs' in pred_data:
                seg_npz = pred_data['segs'].astype(np.uint8)
            elif 'arr_0' in pred_data:
                seg_npz = pred_data['arr_0'].astype(np.uint8)
            else:
                # Try to guess or take the first key
                seg_npz = pred_data[list(pred_data.keys())[0]].astype(np.uint8)
                
            # Image Metadata
            img_npz = np.load(img_file_path, allow_pickle=True)
            spacing = img_npz['spacing']
            prompts = img_npz['text_prompts']
            if hasattr(prompts, 'item'):
                prompts = prompts.item()
            
            instance_label = prompts.get('instance_label', 0)
            class_ids = sorted([int(k) for k in prompts if k != "instance_label"])
            class_ids_array = np.array(class_ids, dtype=np.int32)
            gt_npz_re,_,_ = process_input(gt_npz,256, mode='nearest')
            gt_npz = gt_npz_re
            if gt_npz.shape != seg_npz.shape:
                print(f"Shape Mismatch! GT: {gt_npz.shape}, Pred: {seg_npz.shape}")
                # Optional: resize or skip. For strict eval, we skip or set NaN
                raise ValueError("Shape mismatch between GT and Prediction")
            # 2. Compute Metrics
            dsc = None
            nsd = None
            f1_score = None
            dsc_tp = None

            if instance_label == 0:  # Semantic Segmentation
                dsc = compute_multi_class_dsc(gt_npz, seg_npz, class_ids_array)
                nsd = compute_multi_class_nsd(gt_npz, seg_npz, spacing, class_ids_array)
                f1_score = np.nan
                dsc_tp = np.nan
            
            elif instance_label == 1: # Instance Segmentation
                # Check if we need to convert binary mask to instance mask
                if len(np.unique(seg_npz)) <= 2 and np.max(seg_npz) == 1:
                    # convert prediction masks from binary to instance
                    tumor_inst, _ = cc3d.connected_components(seg_npz, connectivity=6, return_N=True)
                    seg_npz = tumor_inst.astype(np.uint8)
                
                # Relabel sequentially to ensure contiguous IDs for linear_sum_assignment
                gt_npz, _, _ = segmentation.relabel_sequential(gt_npz)
                # F1 / Precision / Recall
                tp, fp, fn, matched_pairs = eval_tp_fp_fn(gt_npz, seg_npz)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                # DSC for True Positives
                if matched_pairs:
                    dsc_list = []
                    for gt_idx, pred_idx in matched_pairs:
                        gt_mask = gt_npz == (gt_idx + 1)
                        pred_mask = seg_npz == (pred_idx + 1)
                        dsc_value = compute_dice_coefficient(gt_mask, pred_mask)
                        dsc_list.append(dsc_value)
                    dsc_tp = np.mean(dsc_list)
                else:
                    dsc_tp = 0.0
                
                dsc = np.nan
                nsd = np.nan

            # 3. Store Results
            metric['CaseName'].append(case)
            metric['DSC'].append(round(dsc, 4) if dsc is not None and not np.isnan(dsc) else np.nan)
            metric['NSD'].append(round(nsd, 4) if nsd is not None and not np.isnan(nsd) else np.nan)
            metric['F1'].append(round(f1_score, 4) if f1_score is not None and not np.isnan(f1_score) else np.nan)
            metric['DSC_TP'].append(round(dsc_tp, 4) if dsc_tp is not None and not np.isnan(dsc_tp) else np.nan)

            print(f"Done. DSC={metric['DSC'][-1]}, NSD={metric['NSD'][-1]}, F1={metric['F1'][-1]}")

        except Exception as e:
            print(f"ERROR: {e}")
            metric['CaseName'].append(case)
            metric['DSC'].append(np.nan)
            metric['NSD'].append(np.nan)
            metric['F1'].append(np.nan)
            metric['DSC_TP'].append(np.nan)
            missing_files.append(f"{case}: {str(e)}")

    # 4. Save Final CSV
    metric_df = pd.DataFrame(metric)
    csv_filename = join(save_path, f'{teamname}_metrics.csv')
    metric_df.to_csv(csv_filename, index=False)
    print(f"\nEvaluation Complete. Metrics saved to: {csv_filename}")
    
    # Compute and print averages
    print("-" * 30)
    print("Summary:")
    print(f"Mean DSC: {metric_df['DSC'].mean():.4f}")
    print(f"Mean NSD: {metric_df['NSD'].mean():.4f}")
    print(f"Mean F1:  {metric_df['F1'].mean():.4f}")

    if missing_files:
        error_log_path = join(save_path, f'{teamname}_errors.txt')
        with open(error_log_path, 'w') as f:
            f.write("\n".join(missing_files))
        print(f"WARNING: {len(missing_files)} cases failed. See {error_log_path}")

if __name__ == "__main__":
    main()