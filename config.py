# config.py
import os

# Base paths (update to your Google Drive mount point)
DRIVE_PATH = '/content/drive/MyDrive/'

# Input files
STORE_MASTER = os.path.join(DRIVE_PATH, 'store_master.csv')
EMPLOYEE_MASTER = os.path.join(DRIVE_PATH, 'employee_master.csv')
TRANSACTIONS = os.path.join(DRIVE_PATH, 'transactions.csv')
FEATURES_EMPLOYEE_MONTHLY = os.path.join(DRIVE_PATH, 'features_employee_monthly.csv')
FEATURES_POS_MONTHLY = os.path.join(DRIVE_PATH, 'features_pos_monthly.csv')
ANOMALY_EMPLOYEES = os.path.join(DRIVE_PATH, 'anomaly_employees.csv')  # ground truth

# Output files
OUTPUT_EMPLOYEE_RISK = os.path.join(DRIVE_PATH, 'employee_risk.csv')
OUTPUT_POS_RISK = os.path.join(DRIVE_PATH, 'pos_risk.csv')
OUTPUT_PATTERNS = os.path.join(DRIVE_PATH, 'repeat_patterns.json')

# Column names – ADJUST THESE TO MATCH YOUR CSV HEADERS
COL_EMPLOYEE_ID = 'employee_id'
COL_STORE_ID = 'store_id'
COL_POS_TERMINAL_ID = 'pos_terminal_id'
COL_TRANSACTION_DATETIME = 'transaction_datetime'   # <-- CHANGE to actual column name (e.g., 'tran_date')
COL_SALE_AMOUNT = 'sale_amount'
COL_DISCOUNT_AMOUNT = 'discount_amount'
COL_VOID_FLAG = 'void_flag'
COL_REFUND_FLAG = 'refund_flag'
COL_REFUND_AMOUNT = 'refund_amount'
COL_OVERRIDE_FLAG = 'override_flag'
COL_PAYMENT_METHOD = 'payment_method'
COL_RECEIPT_PROVIDED = 'receipt_provided'

# Detection parameters
ROLLING_WINDOW_DAYS = 30
ZSCORE_THRESHOLD = 2.5
IQR_MULTIPLIER = 1.5
RISK_WEIGHTS = {
    'refund_anomaly': 0.3,
    'void_anomaly': 0.2,
    'override_anomaly': 0.2,
    'cluster_risk': 0.2,
    'time_anomaly': 0.1
}
RISK_THRESHOLDS = {
    'high': 85,
    'moderate': 70
}

# Validation targets
TARGET_RECALL = 0.75
TARGET_FPR = 0.10