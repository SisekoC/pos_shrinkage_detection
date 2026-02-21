# diagnose_files.py
import pandas as pd
from config import *

files = {
    'STORE_MASTER': STORE_MASTER,
    'EMPLOYEE_MASTER': EMPLOYEE_MASTER,
    'TRANSACTIONS': TRANSACTIONS,
    'FEATURES_EMPLOYEE_MONTHLY': FEATURES_EMPLOYEE_MONTHLY,
    'FEATURES_POS_MONTHLY': FEATURES_POS_MONTHLY,
    'ANOMALY_EMPLOYEES': ANOMALY_EMPLOYEES
}

for name, path in files.items():
    print(f"\n--- {name} ---")
    try:
        df = pd.read_csv(path)
        print(f"Shape: {df.shape}")
        print("Columns:", list(df.columns))
        print("First row:", df.iloc[0].to_dict())
    except Exception as e:
        print(f"Error: {e}")