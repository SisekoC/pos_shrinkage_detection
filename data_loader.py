# data_loader.py
import pandas as pd
import os
from config import *

class DataLoader:
    @staticmethod
    def load_store_master():
        return pd.read_csv(STORE_MASTER)

    @staticmethod
    def load_employee_master():
        return pd.read_csv(EMPLOYEE_MASTER)

    @staticmethod
    def load_transactions():
        # Ensure datetime parsing
        df = pd.read_csv(TRANSACTIONS, parse_dates=['transaction_datetime'])
        return df

    @staticmethod
    def load_features_employee():
        return pd.read_csv(FEATURES_EMPLOYEE_MONTHLY)

    @staticmethod
    def load_features_pos():
        return pd.read_csv(FEATURES_POS_MONTHLY)

    @staticmethod
    def load_anomaly_employees():
        # Expected columns: employee_id, ground_truth (1 if fraud actor)
        return pd.read_csv(ANOMALY_EMPLOYEES)

    @staticmethod
    def load_all():
        return {
            'store_master': DataLoader.load_store_master(),
            'employee_master': DataLoader.load_employee_master(),
            'transactions': DataLoader.load_transactions(),
            'features_employee': DataLoader.load_features_employee(),
            'features_pos': DataLoader.load_features_pos(),
            'anomaly_employees': DataLoader.load_anomaly_employees()
        }