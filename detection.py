# detection.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import *

class BehavioralCluster:
    """Layer 2: Cluster employees based on behavior patterns."""
    
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = None
        self.scaler = StandardScaler()

    def fit_predict(self, features_df, feature_columns):
        X = features_df[feature_columns].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = self.model.fit_predict(X_scaled)
        
        # Compute cluster risk based on centroid values
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        cluster_risk = {}
        for i, center in enumerate(centers):
            # Use first three features as proxies for risk (refund, void, override)
            risk = np.mean(center[:3]) if len(center) >= 3 else 0.5
            cluster_risk[i] = risk
        
        features_df['cluster'] = labels
        features_df['cluster_risk'] = features_df['cluster'].map(cluster_risk)
        # Normalize cluster_risk to 0-1
        min_r = features_df['cluster_risk'].min()
        max_r = features_df['cluster_risk'].max()
        if max_r > min_r:
            features_df['cluster_risk'] = (features_df['cluster_risk'] - min_r) / (max_r - min_r)
        else:
            features_df['cluster_risk'] = 0.5
        return features_df


class PatternDetector:
    """Layer 3: Detect specific repeat patterns from transaction data."""
    
    def __init__(self, features_employee):
        self.features = features_employee

    def detect_high_discount_cash(self, discount_threshold=0.3):
        """Count high discount + cash transactions per employee."""
        if COL_DISCOUNT_AMOUNT not in self.transactions.columns or COL_SALE_AMOUNT not in self.transactions.columns:
            return pd.Series(dtype=int)
        self.transactions['discount_pct'] = self.transactions[COL_DISCOUNT_AMOUNT] / self.transactions[COL_SALE_AMOUNT].clip(1)
        if COL_PAYMENT_METHOD not in self.transactions.columns:
            return pd.Series(dtype=int)
        mask = (self.transactions['discount_pct'] > discount_threshold) & (self.transactions[COL_PAYMENT_METHOD] == 'cash')
        return self.transactions[mask].groupby(COL_EMPLOYEE_ID).size()

    def detect_refund_no_receipt(self):
        """Count refunds without receipt per employee."""
        if COL_REFUND_FLAG not in self.transactions.columns or COL_RECEIPT_PROVIDED not in self.transactions.columns:
            return pd.Series(dtype=int)
        mask = (self.transactions[COL_REFUND_FLAG] == 1) & (self.transactions[COL_RECEIPT_PROVIDED] == 0)
        return self.transactions[mask].groupby(COL_EMPLOYEE_ID).size()

    def get_pattern_counts(self):
        # Assume features contains columns: high_discount_cash_count, refund_no_receipt_count
        # Use the most recent month for each employee
        latest = self.features.sort_values(['employee_id', 'year_month']).groupby('employee_id').last()
        return latest[['high_discount_cash_count', 'refund_no_receipt_count']].fillna(0)

class CompositeRiskScorer:
    """Layer 4: Combine all signals into a single risk score."""
    
    def __init__(self, weights=None, thresholds=None):
        self.weights = weights or RISK_WEIGHTS
        self.thresholds = thresholds or RISK_THRESHOLDS

    def score_employees(self, features_df, pattern_counts):
        """
        features_df: employee features with z-scores, cluster_risk, etc.
        pattern_counts: DataFrame with pattern counts per employee (index=employee_id)
        Returns DataFrame with risk_score, flags, reason codes.
        """
        # Normalize anomaly components (z-scores clipped to 0-3 then scaled to 0-1)
        refund_anomaly = np.clip(features_df.get('refund_rate_zscore', 0).abs(), 0, 3) / 3
        void_anomaly = np.clip(features_df.get('void_rate_zscore', 0).abs(), 0, 3) / 3
        override_anomaly = np.clip(features_df.get('override_rate_zscore', 0).abs(), 0, 3) / 3
        
        # Cluster risk
        cluster_risk = features_df.get('cluster_risk', 0).fillna(0)
        
        # Time anomaly
        time_anomaly = features_df.get('time_anomaly', 0)
        
        # Weighted sum
        raw_score = (self.weights['refund_anomaly'] * refund_anomaly +
                     self.weights['void_anomaly'] * void_anomaly +
                     self.weights['override_anomaly'] * override_anomaly +
                     self.weights['cluster_risk'] * cluster_risk +
                     self.weights['time_anomaly'] * time_anomaly)
        
        risk_score = raw_score * 100
        
        # Reason codes
        contributions = pd.DataFrame({
            'refund': refund_anomaly * self.weights['refund_anomaly'] * 100,
            'void': void_anomaly * self.weights['void_anomaly'] * 100,
            'override': override_anomaly * self.weights['override_anomaly'] * 100,
            'cluster': cluster_risk * self.weights['cluster_risk'] * 100,
            'time': time_anomaly * self.weights['time_anomaly'] * 100
        })
        reason_code = contributions.idxmax(axis=1)
        
        # Risk flag
        risk_flag = pd.cut(risk_score,
                           bins=[0, self.thresholds['moderate'], self.thresholds['high'], 100],
                           labels=['Monitor', 'Moderate', 'High'],
                           right=False)
        
        # Build output
        output = features_df[[COL_EMPLOYEE_ID, COL_STORE_ID]].copy()
        output['risk_score'] = risk_score
        output['refund_rate'] = features_df.get('refund_rate', 0)
        output['void_rate'] = features_df.get('void_rate', 0)
        output['override_rate'] = features_df.get('override_rate', 0)
        output['peer_percentile'] = features_df.get('refund_rate_percentile', 0) * 100
        output['risk_flag'] = risk_flag
        output['reason_code'] = reason_code
        
        # Merge pattern counts
        output = output.merge(pattern_counts, left_on=COL_EMPLOYEE_ID, right_index=True, how='left').fillna(0)
        return output

    def score_terminals(self, features_pos, employee_risk):
        """
        features_pos: POS monthly features
        employee_risk: result from score_employees (contains employee risk scores)
        """
        # Ensure required columns exist
        if 'high_risk_employee_ratio' not in features_pos.columns:
            features_pos['high_risk_employee_ratio'] = 0.0
        
        # Compute terminal risk score
        features_pos['risk_score'] = (
            features_pos.get('refund_rate', 0) * 20 +
            features_pos.get('void_rate', 0) * 20 +
            features_pos.get('override_rate', 0) * 20 +
            features_pos['high_risk_employee_ratio'] * 40
        ) * 100
        
        features_pos['risk_flag'] = pd.cut(features_pos['risk_score'],
                                           bins=[0, 70, 85, 100],
                                           labels=['Monitor', 'Moderate', 'High'],
                                           right=False)
        # Select output columns
        cols = [COL_POS_TERMINAL_ID, COL_STORE_ID, 'risk_score', 'refund_rate', 'void_rate', 
                'high_risk_employee_ratio', 'risk_flag']
        cols = [c for c in cols if c in features_pos.columns]
        return features_pos[cols]