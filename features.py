# features.py
import pandas as pd
import numpy as np
from config import *

class FeatureProcessor:
    def __init__(self, features_employee, features_pos, transactions, employee_master):
        self.features_employee = features_employee.copy()
        self.features_pos = features_pos.copy()
        self.transactions = transactions
        self.employee_master = employee_master

    def add_peer_comparisons(self):
        """
        For each employee metric, compute z-score and percentile relative to store peers.
        Assumes features_employee has columns: employee_id, store_id, and metric columns.
        """
        metrics = ['refund_rate', 'void_rate', 'override_rate', 'avg_discount_pct', 'cash_rate', 'no_receipt_refund_rate']
        # Ensure these columns exist; if not, we'll compute from provided features.
        # For demonstration, we'll assume they exist; adjust based on actual column names.
        
        # Merge store_id from employee_master if not present
        if 'store_id' not in self.features_employee.columns:
            self.features_employee = self.features_employee.merge(
                self.employee_master[['employee_id', 'store_id']], on='employee_id', how='left'
            )
        
        for metric in metrics:
            if metric not in self.features_employee.columns:
                continue
            # Compute store-level stats
            store_stats = self.features_employee.groupby('store_id')[metric].agg(['mean', 'std']).rename(
                columns={'mean': f'{metric}_store_mean', 'std': f'{metric}_store_std'}
            ).fillna(0)
            self.features_employee = self.features_employee.merge(store_stats, on='store_id', how='left')
            self.features_employee[f'{metric}_zscore'] = (
                self.features_employee[metric] - self.features_employee[f'{metric}_store_mean']
            ) / self.features_employee[f'{metric}_store_std'].clip(1e-6)
            
            # Percentile within store
            self.features_employee[f'{metric}_percentile'] = self.features_employee.groupby('store_id')[metric].rank(pct=True)
        
        return self.features_employee

    def add_outlier_flags(self):
        """Add z-score, IQR, percentile flags based on config thresholds."""
        metrics = [col for col in self.features_employee.columns if col.endswith('_zscore')]
        for col in metrics:
            base = col.replace('_zscore', '')
            # Z-score outlier
            self.features_employee[f'{base}_outlier_z'] = (np.abs(self.features_employee[col]) > ZSCORE_THRESHOLD).astype(int)
            
            # IQR outlier (requires original metric)
            orig = base
            if orig in self.features_employee.columns:
                Q1 = self.features_employee[orig].quantile(0.25)
                Q3 = self.features_employee[orig].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - IQR_MULTIPLIER * IQR
                upper = Q3 + IQR_MULTIPLIER * IQR
                self.features_employee[f'{base}_outlier_iqr'] = (
                    (self.features_employee[orig] < lower) | (self.features_employee[orig] > upper)
                ).astype(int)
            
            # Percentile outlier (top 5%)
            if f'{base}_percentile' in self.features_employee.columns:
                self.features_employee[f'{base}_outlier_pct'] = (self.features_employee[f'{base}_percentile'] > 0.95).astype(int)
        
        return self.features_employee

    def add_time_anomaly(self):
        """
        Compute time-of-day anomaly score per employee.
        This requires transaction-level data to analyze hour-of-day patterns.
        For simplicity, we'll create a placeholder.
        """
        # Placeholder: can be computed from self.transactions grouped by employee_id
        # e.g., transactions after 10pm, weekend concentration, etc.
        # For now, set to 0.
        self.features_employee['time_anomaly'] = 0.0
        return self.features_employee

    def process(self):
        self.features_employee = self.add_peer_comparisons()
        self.features_employee = self.add_outlier_flags()
        self.features_employee = self.add_time_anomaly()
        return self.features_employee, self.features_pos