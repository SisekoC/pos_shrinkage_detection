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
        # Define metrics that should be present (adjust based on actual column names)
        possible_metrics = ['refund_rate', 'void_rate', 'override_rate', 'avg_discount_pct', 'cash_rate', 'no_receipt_refund_rate']
        metrics = [m for m in possible_metrics if m in self.features_employee.columns]
        
        # Ensure store_id is present
        if COL_STORE_ID not in self.features_employee.columns:
            self.features_employee = self.features_employee.merge(
                self.employee_master[[COL_EMPLOYEE_ID, COL_STORE_ID]], 
                on=COL_EMPLOYEE_ID, how='left'
            )
        
        for metric in metrics:
            # Compute store-level stats
            store_stats = self.features_employee.groupby(COL_STORE_ID)[metric].agg(['mean', 'std']).rename(
                columns={'mean': f'{metric}_store_mean', 'std': f'{metric}_store_std'}
            ).fillna(0)
            self.features_employee = self.features_employee.merge(store_stats, on=COL_STORE_ID, how='left')
            self.features_employee[f'{metric}_zscore'] = (
                self.features_employee[metric] - self.features_employee[f'{metric}_store_mean']
            ) / self.features_employee[f'{metric}_store_std'].clip(1e-6)
            
            # Percentile within store
            self.features_employee[f'{metric}_percentile'] = self.features_employee.groupby(COL_STORE_ID)[metric].rank(pct=True)
        
        return self.features_employee

    def add_outlier_flags(self):
        """Add z-score, IQR, percentile flags based on config thresholds."""
        metrics = [col.replace('_zscore', '') for col in self.features_employee.columns if col.endswith('_zscore')]
        for base in metrics:
            # Z-score outlier
            zcol = f'{base}_zscore'
            if zcol in self.features_employee.columns:
                self.features_employee[f'{base}_outlier_z'] = (np.abs(self.features_employee[zcol]) > ZSCORE_THRESHOLD).astype(int)
            
            # IQR outlier (requires original metric)
            if base in self.features_employee.columns:
                Q1 = self.features_employee[base].quantile(0.25)
                Q3 = self.features_employee[base].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - IQR_MULTIPLIER * IQR
                upper = Q3 + IQR_MULTIPLIER * IQR
                self.features_employee[f'{base}_outlier_iqr'] = (
                    (self.features_employee[base] < lower) | (self.features_employee[base] > upper)
                ).astype(int)
            
            # Percentile outlier (top 5%)
            pctcol = f'{base}_percentile'
            if pctcol in self.features_employee.columns:
                self.features_employee[f'{base}_outlier_pct'] = (self.features_employee[pctcol] > 0.95).astype(int)
        
        return self.features_employee

    def add_time_anomaly(self):
        if 'late_night_txn_pct' in self.features_employee.columns:
            # Use percentile rank as time anomaly score
            self.features_employee['time_anomaly'] = self.features_employee.groupby('store_id')['late_night_txn_pct'].rank(pct=True)
        else:
            self.features_employee['time_anomaly'] = 0.0
        return self.features_employee

    def process(self):
        self.features_employee = self.add_peer_comparisons()
        self.features_employee = self.add_outlier_flags()
        self.features_employee = self.add_time_anomaly()
        return self.features_employee, self.features_pos