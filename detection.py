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
        
        # Compute cluster risk based on centroid values (higher refund/void = higher risk)
        centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        # Assume feature_columns order matches; compute a simple risk score for each cluster
        # For demonstration, we'll use the mean of the first three features (refund, void, override)
        cluster_risk = {}
        for i, center in enumerate(centers):
            # Normalize each feature to 0-1 based on overall max/min? Or just use rank.
            # Simple: risk = average of normalized refund, void, override rates
            # We'll assume feature_columns[0] = refund_rate, [1] = void_rate, [2] = override_rate
            risk = np.mean(center[:3])  # crude, but works for demo
            cluster_risk[i] = risk
        
        features_df['cluster'] = labels
        features_df['cluster_risk'] = features_df['cluster'].map(cluster_risk)
        # Normalize cluster_risk to 0-1
        features_df['cluster_risk'] = (features_df['cluster_risk'] - features_df['cluster_risk'].min()) / \
                                       (features_df['cluster_risk'].max() - features_df['cluster_risk'].min() + 1e-6)
        return features_df


class PatternDetector:
    """Layer 3: Detect specific repeat patterns from transaction data."""
    
    def __init__(self, transactions):
        self.transactions = transactions

    def detect_high_discount_cash(self, discount_threshold=0.3):
        """Count high discount + cash transactions per employee."""
        self.transactions['discount_pct'] = self.transactions['discount_amount'] / self.transactions['sale_amount'].clip(1)
        mask = (self.transactions['discount_pct'] > discount_threshold) & (self.transactions['payment_method'] == 'cash')
        return self.transactions[mask].groupby('employee_id').size()

    def detect_refund_no_receipt(self):
        """Count refunds without receipt per employee."""
        mask = (self.transactions['refund_flag'] == 1) & (self.transactions['receipt_provided'] == 0)
        return self.transactions[mask].groupby('employee_id').size()

    def detect_sale_void_resale(self, window_hours=1):
        """
        Detect sale -> void -> resale sequences within short time by same employee.
        Simplified: count voids that are followed by a cash sale of similar amount within window.
        """
        # This is complex; for demo, we return empty series.
        return pd.Series(dtype=int)

    def get_pattern_counts(self):
        """Return a DataFrame with pattern counts per employee."""
        hd = self.detect_high_discount_cash()
        nr = self.detect_refund_no_receipt()
        # Combine
        patterns = pd.DataFrame({'high_discount_cash': hd, 'refund_no_receipt': nr}).fillna(0)
        patterns['total_patterns'] = patterns.sum(axis=1)
        return patterns


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
        refund_anomaly = np.clip(features_df['refund_rate_zscore'].abs(), 0, 3) / 3
        void_anomaly = np.clip(features_df['void_rate_zscore'].abs(), 0, 3) / 3
        override_anomaly = np.clip(features_df['override_rate_zscore'].abs(), 0, 3) / 3
        
        # Cluster risk already 0-1
        cluster_risk = features_df['cluster_risk'].fillna(0)
        
        # Time anomaly (placeholder)
        time_anomaly = features_df['time_anomaly'] if 'time_anomaly' in features_df else 0
        
        # Weighted sum
        raw_score = (self.weights['refund_anomaly'] * refund_anomaly +
                     self.weights['void_anomaly'] * void_anomaly +
                     self.weights['override_anomaly'] * override_anomaly +
                     self.weights['cluster_risk'] * cluster_risk +
                     self.weights['time_anomaly'] * time_anomaly)
        
        # Scale to 0-100
        risk_score = raw_score * 100
        
        # Reason codes: top contributing factor
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
        output = features_df[['employee_id', 'store_id']].copy()
        output['risk_score'] = risk_score
        output['refund_rate'] = features_df.get('refund_rate', 0)
        output['void_rate'] = features_df.get('void_rate', 0)
        output['override_rate'] = features_df.get('override_rate', 0)
        output['peer_percentile'] = features_df.get('refund_rate_percentile', 0) * 100  # example
        output['risk_flag'] = risk_flag
        output['reason_code'] = reason_code
        
        # Merge pattern counts
        output = output.merge(pattern_counts, left_on='employee_id', right_index=True, how='left').fillna(0)
        return output

    def score_terminals(self, features_pos, employee_risk):
        """
        features_pos: POS monthly features
        employee_risk: result from score_employees (contains employee risk scores)
        """
        # Need to link employees to terminals via transactions or a mapping.
        # For simplicity, we'll assume features_pos already contains high_risk_employee_ratio
        # If not, we need to compute it. We'll compute if missing.
        if 'high_risk_employee_ratio' not in features_pos.columns:
            # Placeholder: set to 0
            features_pos['high_risk_employee_ratio'] = 0.0
        
        # Compute risk score for terminals
        features_pos['risk_score'] = (
            features_pos.get('refund_rate', 0) * 20 +
            features_pos.get('void_rate', 0) * 20 +
            features_pos.get('override_rate', 0) * 20 +
            features_pos['high_risk_employee_ratio'] * 40
        ) * 100  # scale to 0-100
        
        features_pos['risk_flag'] = pd.cut(features_pos['risk_score'],
                                           bins=[0, 70, 85, 100],
                                           labels=['Monitor', 'Moderate', 'High'],
                                           right=False)
        # Ensure required columns
        cols = ['pos_terminal_id', 'store_id', 'risk_score', 'refund_rate', 'void_rate', 
                'high_risk_employee_ratio', 'risk_flag']
        return features_pos[[c for c in cols if c in features_pos.columns]]