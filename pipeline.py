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
    
    transactions = data['transactions']
    features_employee = data['features_employee']
    features_pos = data['features_pos']
    employee_master = data['employee_master']
    anomaly_employees = data['anomaly_employees']  # ground truth
    
    # Process features: add peer comparisons, outlier flags, time anomaly
    print("Processing features...")
    fp = FeatureProcessor(features_employee, features_pos, transactions, employee_master)
    features_employee, features_pos = fp.process()
    
    # Layer 2: Behavioral clustering
    print("Clustering employees...")
    cluster = BehavioralCluster(n_clusters=5)
    cluster_features = ['refund_rate', 'void_rate', 'override_rate', 'avg_discount_pct', 'cash_rate']
    # Ensure columns exist; if not, adjust
    cluster_features = [c for c in cluster_features if c in features_employee.columns]
    features_employee = cluster.fit_predict(features_employee, cluster_features)
    
    # Layer 3: Pattern detection
    print("Detecting patterns...")
    pattern_detector = PatternDetector(transactions)
    pattern_counts = pattern_detector.get_pattern_counts()
    
    # Layer 4: Composite scoring
    print("Computing risk scores...")
    scorer = CompositeRiskScorer()
    employee_risk = scorer.score_employees(features_employee, pattern_counts)
    pos_risk = scorer.score_terminals(features_pos, employee_risk)
    
    # Validation if ground truth exists
    if anomaly_employees is not None and 'employee_id' in anomaly_employees.columns:
        print("Validating against ground truth...")
        # Merge ground truth with employee_risk
        eval_df = employee_risk.merge(anomaly_employees, on='employee_id', how='left')
        y_true = eval_df['ground_truth'].fillna(0)
        y_scores = eval_df['risk_score']
        
        validator = Validator()
        best_thresh, best_metrics = validator.find_best_threshold(y_true, y_scores)
        if best_thresh:
            print(f"Optimal threshold found: {best_thresh:.3f}")
            print(f"  Precision: {best_metrics[0]:.3f}, Recall: {best_metrics[1]:.3f}, FPR: {best_metrics[2]:.3f}")
            # Apply threshold to risk_flag
            employee_risk['risk_flag'] = (y_scores >= best_thresh).map({True: 'High', False: 'Monitor'})
        else:
            print("No threshold meets targets. Performing error analysis with default thresholds...")
            # Use config thresholds
            error_analysis = validator.error_analysis(y_true, y_scores, RISK_THRESHOLDS['high'], 
                                                      eval_df['employee_id'], 
                                                      fraud_types=anomaly_employees.get('fraud_type'))
            print(f"False negatives: {error_analysis['fn_count']}, False positives: {error_analysis['fp_count']}")
            if 'false_negatives_by_type' in error_analysis:
                print("Missed fraud types:", error_analysis['false_negatives_by_type'])
    else:
        print("No ground truth provided; skipping validation.")
    
    # Generate repeat pattern report (top 5 per store)
    print("Generating repeat pattern report...")
    # Aggregate pattern counts by store (requires joining employee to store)
    pattern_with_store = pattern_counts.reset_index().merge(
        employee_master[['employee_id', 'store_id']], on='employee_id', how='left'
    )
    # For each store, get top patterns
    report = {}
    for store_id in pattern_with_store['store_id'].unique():
        store_patterns = pattern_with_store[pattern_with_store['store_id'] == store_id]
        # Sum patterns per store
        totals = store_patterns[['high_discount_cash', 'refund_no_receipt']].sum()
        top = totals.sort_values(ascending=False).head(5)
        report[int(store_id)] = {
            'top_patterns': top.to_dict(),
            'total_patterns': int(totals.sum())
        }
    
    # Save outputs
    print("Saving outputs...")
    employee_risk.to_csv(OUTPUT_EMPLOYEE_RISK, index=False)
    pos_risk.to_csv(OUTPUT_POS_RISK, index=False)
    with open(OUTPUT_PATTERNS, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Done. Outputs saved to:", DRIVE_PATH)

if __name__ == "__main__":
    main()