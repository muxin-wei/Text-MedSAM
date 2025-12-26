import torch
import numpy as np
import cc3d
from torchmetrics import Metric
from evaluation.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from skimage import segmentation
from scipy.ndimage import distance_transform_edt, binary_closing, generate_binary_structure
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional.segmentation import dice_score

class SegEval(Metric):
    def __init__(
        self,
        iou_threshold = 0.5,
        dist_sync_on_step=False
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.iou_threshold = iou_threshold
        self.nsd_tolerance = 2.0
        # instance seg
        self.add_state("F1", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("TP", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("FP", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("FN", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("DSC_TP", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("num_instance", default=torch.tensor(0).long(), dist_reduce_fx="sum")
        
        # semantic seg
        self.add_state("DSC", default=torch.tensor(0.), dist_reduce_fx="sum")
        # self.add_state("NSD", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("num_semantics", default=torch.tensor(0).long(), dist_reduce_fx="sum")
        
    def update(self, pred_mask, gt_mask, cls_ids):
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.cpu().numpy().astype(np.int32)
            gt_mask = gt_mask.cpu().numpy().astype(np.int32)
            self._process_sample(pred_mask=pred_mask, gt_mask=gt_mask, cls_ids=cls_ids)
    
    def compute(self):
        # nsd = self.NSD / self.num_semantics if self.num_semantics > 0 else 0.0
        dsc = self.DSC / self.num_semantics if self.num_semantics > 0 else 0.0
        
        dsc_tp = self.DSC_TP / self.num_instance if self.num_instance > 0 else 0.0
        f1 = self.F1 / self.num_instance if self.num_instance > 0 else 0.0
        tp = self.TP / self.num_instance if self.num_instance > 0 else 0.0
        fp = self.FP / self.num_instance if self.num_instance > 0 else 0.0
        fn = self.FN / self.num_instance if self.num_instance > 0 else 0.0
        
        return {
            # "nsd": nsd,
            "dsc": dsc,
            "dsc_tp": dsc_tp,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        } 
    
    def _process_sample(self, pred_mask, gt_mask, cls_ids):
        if len(cls_ids) == 1: # compute F1_50, DSC_TP
            tumor_inst, tumor_n = cc3d.connected_components(pred_mask, connectivity=6, return_N=True)
            pred_mask[tumor_inst > 0] = (tumor_inst[tumor_inst > 0] + np.max(pred_mask)) # tumour label + 1
            pred_mask = segmentation.relabel_sequential(pred_mask)[0]
            gt_mask = segmentation.relabel_sequential(gt_mask)[0]
            # F1 COMPUTATION
            tp, fp, fn, matched_pairs = self._eval_tp_fp_fn(gt_mask, pred_mask, self.iou_threshold)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            # DSC_TP
            dsc_tp = 0.
            if len(matched_pairs) > 0:
                dsc_tp = self._compute_matched_dice(gt_mask, pred_mask, matched_pairs)
    
            self.DSC_TP += float(dsc_tp)
            self.TP += float(tp)
            self.FP += float(fp)
            self.FN += float(fn)
            self.F1 += float(f1)
            self.num_instance += 1
        else:
            present_labels = set(np.unique(gt_mask)[1:]) & set(cls_ids)
            present_labels = list(present_labels)
            if not present_labels:
                return
            mean_dsc = self._compute_multi_class_dsc(gt=gt_mask, seg=pred_mask, label_ids=present_labels)
            # mean_nsd = self._compute_multi_class_nsd(gt=gt_mask, seg=pred_mask, label_ids=present_labels, tolerance=self.nsd_tolerance)
            self.DSC += float(mean_dsc)
            # self.NSD += float(mean_nsd)
            self.num_semantics +=1
                
    #############################################################
    #                                                           #
    #                          helpers                          #
    #                                                           #
    #############################################################
    
    def _eval_tp_fp_fn(self, masks_true, masks_pred, threshold=0.5):
        num_inst_gt = np.max(masks_true)
        num_inst_seg = np.max(masks_pred)
        if num_inst_seg>0:
            iou = self._intersection_over_union(masks_true, masks_pred)[1:, 1:]
            tp, matched_pairs = self._true_positive(iou, threshold)
            fp = num_inst_seg - tp
            fn = num_inst_gt - tp
        else:
            # print('No segmentation results!')
            tp = 0
            fp = 0
            fn = num_inst_gt
            matched_pairs = []
            
        return tp, fp, fn, matched_pairs
    
    def _compute_matched_dice(self, gt, seg, matched_pairs):
        dsc_list = [0.]
        for gt_idx, pred_idx in matched_pairs:
            gt_mask_i = (gt == (gt_idx + 1))
            pred_mask_i = (seg == (pred_idx + 1))
            dsc_value = compute_dice_coefficient(gt_mask_i, pred_mask_i)
            dsc_list.append(dsc_value)
        return np.nanmean(dsc_list)
    
    
    def _compute_multi_class_dsc(self, gt, seg, label_ids):
        dsc = [0.] * len(label_ids)
        for idx, i in enumerate(label_ids):
            gt_i = gt == i
            seg_i = seg == i
            dsc[idx] = compute_dice_coefficient(gt_i, seg_i)
        if not dsc:
            return 0.
        return np.nanmean(dsc)
    
    # def _compute_multi_class_nsd(self, gt, seg, label_ids, spacing=(1.0, 1.0, 1.0), tolerance=2.0):
    #     if not label_ids:
    #         return np.nan
    #     nsd = []
    #     for i in label_ids:
    #         gt_i = gt == i
    #         seg_i = seg == i
    #         if np.sum(gt_i) == 0 and np.sum(seg_i) == 0:
    #             nsd.append(1.0)
    #             continue
    #         if np.sum(gt_i) == 0 or np.sum(seg_i) == 0:
    #             nsd.append(0.0)
    #             continue
    #         surface_distance = compute_surface_distances(gt_i, seg_i, spacing_mm=spacing)
    #         nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    #     return np.nanmean(nsd)
    
    def _label_overlap(self, x, y):
        x = x.ravel()
        y = y.ravel()
        n_class_x = x.max() + 1
        n_class_y = y.max() + 1
        
        max_val = n_class_x * n_class_y
        if max_val < 2147483647: # int32 max
            dtype = np.int32
        else:
            dtype = np.int64
            
        flat_ids = x.astype(dtype) * n_class_y + y.astype(dtype)
        counts = np.bincount(flat_ids, minlength=n_class_x * n_class_y)
        overlap = counts.reshape((n_class_x, n_class_y))
        return overlap
    
    def _intersection_over_union(self, gt, pred):
        overlap = self._label_overlap(gt, pred)
        n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
        n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
        union = n_pixels_pred + n_pixels_true - overlap
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = overlap / union
        iou[np.isnan(iou)] = 0.
        return iou
    
    def _true_positive(self, iou, thresh_hold):
        n_min = min(iou.shape[0], iou.shape[1])
        if n_min == 0:
            return 0, []
        costs = -(iou >= thresh_hold).astype(float) - iou / (2*n_min)
        true_ind, pred_ind = linear_sum_assignment(costs)
        match_ok = iou[true_ind, pred_ind] >= thresh_hold
        tp = match_ok.sum()
        matched_pairs = [(t, p) for t, p, ok in zip(true_ind, pred_ind, match_ok) if ok]
        return tp, matched_pairs