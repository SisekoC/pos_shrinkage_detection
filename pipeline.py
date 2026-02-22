# pipeline.py
import pandas as pd
import json
import sys
import traceback
from data_loader import DataLoader
from features import FeatureProcessor
from detection import BehavioralCluster, PatternDetector, CompositeRiskScorer
from validation import Validator
from config import *

def main():
    print("=" * 50)
    print("Starting shrinkage detection pipeline")
    print("=" * 50)

    # 1. Load data
    print("\n[1] Loading data...")
    try:
        data = DataLoader.load_all()
        print("   ✓ Data loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Unpack data (transactions is no longer used)
    features_employee = data.get('features_employee')
    features_pos = data.get('features_pos')
    employee_master = data.get('employee_master')
    anomaly_employees = data.get('anomaly_employees')

    # Check essential data
    if features_employee is None or features_employee.empty:
        print("   ✗ features_employee is missing or empty")
        sys.exit(1)
    print(f"   ✓ Employee features shape: {features_employee.shape}")

    if features_pos is None or features_pos.empty:
        print("   ✗ features_pos is missing or empty")
        sys.exit(1)
    print(f"   ✓ POS features shape: {features_pos.shape}")

    if employee_master is None or employee_master.empty:
        print("   ✗ employee_master is missing or empty")
        sys.exit(1)
    print(f"   ✓ Employee master shape: {employee_master.shape}")

    if anomaly_employees is not None:
        print(f"   ✓ Ground truth loaded: {anomaly_employees.shape}")
    else:
        print("   ℹ No ground truth provided")

    # 2. Process features: add peer comparisons, outlier flags, time anomaly
    print("\n[2] Processing features...")
    try:
        fp = FeatureProcessor(features_employee, features_pos, None, employee_master)  # transactions = None
        features_employee, features_pos = fp.process()
        print("   ✓ Feature processing complete")
    except Exception as e:
        print(f"   ✗ Error in feature processing: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3. Layer 2: Behavioral clustering
    print("\n[3] Performing behavioral clustering...")
    try:
        cluster = BehavioralCluster(n_clusters=5)
        cluster_features = ['refund_rate', 'void_rate', 'override_rate', 'avg_discount_pct', 'cash_rate']
        cluster_features = [c for c in cluster_features if c in features_employee.columns]
        if cluster_features:
            features_employee = cluster.fit_predict(features_employee, cluster_features)
            print("   ✓ Clustering complete")
        else:
            print("   ⚠ No clustering features available. Setting cluster_risk to 0.5")
            features_employee['cluster_risk'] = 0.5
    except Exception as e:
        print(f"   ✗ Error in clustering: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 4. Layer 3: Pattern detection (using precomputed feature columns)
    print("\n[4] Detecting patterns from features...")
    try:
        pattern_detector = PatternDetector(features_employee)
        pattern_counts = pattern_detector.get_pattern_counts()
        print(f"   ✓ Pattern counts obtained for {len(pattern_counts)} employees")
        if pattern_counts.empty:
            print("   ⚠ Pattern counts are empty (columns may be missing)")
        else:
            print("   Sample pattern counts:")
            print(pattern_counts.head())
    except Exception as e:
        print(f"   ✗ Error in pattern detection: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 5. Layer 4: Composite risk scoring
    print("\n[5] Computing risk scores...")
    try:
        scorer = CompositeRiskScorer()
        employee_risk = scorer.score_employees(features_employee, pattern_counts)
        pos_risk = scorer.score_terminals(features_pos, employee_risk)
        print("   ✓ Risk scores computed")
        print(f"   Employee risk shape: {employee_risk.shape}")
        print(f"   POS risk shape: {pos_risk.shape}")
    except Exception as e:
        print(f"   ✗ Error in risk scoring: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 6. Validation against ground truth (if available)
    print("\n[6] Validating against ground truth...")
    if anomaly_employees is not None and COL_EMPLOYEE_ID in anomaly_employees.columns:
        try:
            # Merge ground truth with employee_risk
            eval_df = employee_risk.merge(anomaly_employees, on=COL_EMPLOYEE_ID, how='left')
            if 'ground_truth' not in eval_df.columns:
                print("   ⚠ No 'ground_truth' column in anomaly_employees; skipping validation.")
            else:
                y_true = eval_df['ground_truth'].fillna(0)
                y_scores = eval_df['risk_score']

                validator = Validator()
                best_thresh, best_metrics = validator.find_best_threshold(y_true, y_scores)
                if best_thresh:
                    print(f"   ✓ Optimal threshold found: {best_thresh:.3f}")
                    print(f"     Precision: {best_metrics[0]:.3f}, Recall: {best_metrics[1]:.3f}, FPR: {best_metrics[2]:.3f}")
                    # Apply threshold to risk_flag
                    employee_risk['risk_flag'] = (y_scores >= best_thresh).map({True: 'High', False: 'Monitor'})
                else:
                    print("   ⚠ No threshold meets targets. Performing error analysis with default thresholds...")
                    error_analysis = validator.error_analysis(
                        y_true, y_scores, RISK_THRESHOLDS['high'],
                        eval_df[COL_EMPLOYEE_ID],
                        fraud_types=anomaly_employees.get('fraud_type')
                    )
                    print(f"   False negatives: {error_analysis['fn_count']}, False positives: {error_analysis['fp_count']}")
                    if 'false_negatives_by_type' in error_analysis:
                        print("   Missed fraud types:", error_analysis['false_negatives_by_type'])
        except Exception as e:
            print(f"   ✗ Error during validation: {e}")
            traceback.print_exc()
    else:
        print("   ℹ No ground truth provided; skipping validation.")

    # 7. Generate repeat pattern report (top 5 per store)
    print("\n[7] Generating repeat pattern report...")
    try:
        # pattern_counts has index = employee_id, columns = pattern counts
        # Merge with employee_master to get store_id
        pattern_with_store = pattern_counts.reset_index().merge(
            employee_master[[COL_EMPLOYEE_ID, COL_STORE_ID]], on=COL_EMPLOYEE_ID, how='left'
        )
        report = {}
        if not pattern_with_store.empty:
            for store_id in pattern_with_store[COL_STORE_ID].unique():
                store_patterns = pattern_with_store[pattern_with_store[COL_STORE_ID] == store_id]
                totals = store_patterns[['high_discount_cash_count', 'refund_no_receipt_count']].sum()
                top = totals.sort_values(ascending=False).head(5)
                report[int(store_id)] = {
                    'top_patterns': top.to_dict(),
                    'total_patterns': int(totals.sum())
                }
        else:
            print("   ⚠ No pattern data available for report.")
        print("   ✓ Report generated")
    except Exception as e:
        print(f"   ✗ Error generating pattern report: {e}")
        traceback.print_exc()
        report = {}

    # 8. Save outputs
    print("\n[8] Saving outputs...")
    try:
        employee_risk.to_csv(OUTPUT_EMPLOYEE_RISK, index=False)
        print(f"   ✓ Employee risk saved to {OUTPUT_EMPLOYEE_RISK}")

        pos_risk.to_csv(OUTPUT_POS_RISK, index=False)
        print(f"   ✓ POS risk saved to {OUTPUT_POS_RISK}")

        with open(OUTPUT_PATTERNS, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"   ✓ Pattern report saved to {OUTPUT_PATTERNS}")
    except Exception as e:
        print(f"   ✗ Error saving outputs: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()