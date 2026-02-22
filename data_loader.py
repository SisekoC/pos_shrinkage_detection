# data_loader.py
import pandas as pd
from config import *

class DataLoader:
    @staticmethod
    def load_store_master():
        try:
            df = pd.read_csv(STORE_MASTER)
            print(f"Loaded store_master: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading store_master: {e}")
            raise

    @staticmethod
    def load_employee_master():
        try:
            df = pd.read_csv(EMPLOYEE_MASTER)
            print(f"Loaded employee_master: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading employee_master: {e}")
            raise

    @staticmethod
    def load_transactions():
        try:
            df = pd.read_csv(TRANSACTIONS)
            print(f"Loaded transactions: {df.shape}")
            # Convert datetime column if it exists
            if COL_TRANSACTION_DATETIME in df.columns:
                df[COL_TRANSACTION_DATETIME] = pd.to_datetime(df[COL_TRANSACTION_DATETIME])
            else:
                print(f"Warning: Column '{COL_TRANSACTION_DATETIME}' not found in transactions.csv")
                print("Available columns:", list(df.columns))
            return df
        except Exception as e:
            print(f"Error loading transactions: {e}")
            raise

    @staticmethod
    def load_features_employee():
        try:
            df = pd.read_csv(FEATURES_EMPLOYEE_MONTHLY)
            print(f"Loaded features_employee_monthly: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading features_employee: {e}")
            raise

    @staticmethod
    def load_features_pos():
        try:
            df = pd.read_csv(FEATURES_POS_MONTHLY)
            print(f"Loaded features_pos_monthly: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading features_pos: {e}")
            raise

    @staticmethod
    def load_anomaly_employees():
        try:
            df = pd.read_csv(ANOMALY_EMPLOYEES)
            print(f"Loaded anomaly_employees: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading anomaly_employees (optional): {e}")
            return None  # Allow missing ground truth

    @staticmethod
    def load_all():
        return {
            'store_master': DataLoader.load_store_master(),
            'employee_master': DataLoader.load_employee_master(),
            # 'transactions': DataLoader.load_transactions(),  # COMMENTED OUT
            'features_employee': DataLoader.load_features_employee(),
            'features_pos': DataLoader.load_features_pos(),
            'anomaly_employees': DataLoader.load_anomaly_employees()
        }