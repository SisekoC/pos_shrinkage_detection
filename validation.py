# validation.py
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix
from config import *

class Validator:
    def __init__(self, target_recall=TARGET_RECALL, target_fpr=TARGET_FPR):
        self.target_recall = target_recall
        self.target_fpr = target_fpr

    def compute_metrics(self, y_true, y_scores, threshold):
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp+fp) > 0 else 0
        recall = tp / (tp + fn) if (tp+fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
        return precision, recall, fpr

    def find_best_threshold(self, y_true, y_scores):
        """Find threshold that meets recall >= target and fpr <= target."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # thresholds from precision_recall_curve are for decision function; we need to align.
        # We'll iterate over thresholds and compute FPR for each.
        best_thresh = None
        best_metrics = None
        for thresh in thresholds:
            if thresh > 1:  # avoid extreme thresholds
                continue
            y_pred = (y_scores >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            recall_i = tp / (tp + fn) if (tp+fn) > 0 else 0
            fpr_i = fp / (fp + tn) if (fp+tn) > 0 else 0
            if recall_i >= self.target_recall and fpr_i <= self.target_fpr:
                # Found a valid threshold; we can choose the highest threshold for best precision
                # For simplicity, we take the first one (lowest threshold) to maximize recall.
                best_thresh = thresh
                precision_i = tp / (tp + fp) if (tp+fp) > 0 else 0
                best_metrics = (precision_i, recall_i, fpr_i)
                break
        return best_thresh, best_metrics

    def error_analysis(self, y_true, y_scores, threshold, employee_ids, fraud_types=None):
        """
        Identify false negatives and false positives.
        fraud_types: Series with fraud type for each employee (if available).
        """
        y_pred = (y_scores >= threshold).astype(int)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        
        fn_employees = employee_ids[fn_mask]
        fp_employees = employee_ids[fp_mask]
        
        result = {
            'false_negatives': list(fn_employees),
            'false_positives': list(fp_employees),
            'fn_count': len(fn_employees),
            'fp_count': len(fp_employees)
        }
        if fraud_types is not None:
            fn_types = fraud_types[fn_mask].value_counts().to_dict()
            result['false_negatives_by_type'] = fn_types
        return result