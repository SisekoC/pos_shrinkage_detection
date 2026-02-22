# pipeline.py
import pandas as pd
import json
from data_loader import DataLoader
from features import FeatureProcessor
from detection import BehavioralCluster, PatternDetector, CompositeRiskScorer
from validation import Validator
from config import *

def main():
    print("Loading data...")
    data = DataLoader.load_all()
    
    # transactions is no longer used
    features_employee = data['features_employee']
    features_pos = data['features_pos']
    employee_master = data['employee_master']
    anomaly_employees = data['anomaly_employees']  # may be None
    
    # Process features: add peer comparisons, outlier flags, time anomaly
    print("Processing features...")
    fp = FeatureProcessor(features_employee, features_pos, None, employee_master)  # pass None for transactions
    features_employee, features_pos = fp.process()
    
    # Layer 2: Behavioral clustering
    print("Clustering employees...")
    cluster = BehavioralCluster(n_clusters=5)
    cluster_features = ['refund_rate', 'void_rate', 'override_rate', 'avg_discount_pct', 'cash_rate']
    cluster_features = [c for c in cluster_features if c in features_employee.columns]
    if cluster_features:
        features_employee = cluster.fit_predict(features_employee, cluster_features)
    else:
        print("Warning: No clustering features available.")
        features_employee['cluster_risk'] = 0.5
    
    # Layer 3: Pattern detection (now uses features_employee directly)
    print("Detecting patterns...")
    pattern_detector = PatternDetector(features_employee)
    pattern_counts = pattern_detector.get_pattern_counts()
    
    # Layer 4: Composite scoring
    print("Computing risk scores...")
    scorer = CompositeRiskScorer()
    employee_risk = scorer.score_employees(features_employee, pattern_counts)
    pos_risk = scorer.score_terminals(features_pos, employee_risk)
    
    # ... rest unchanged (validation, report, save)