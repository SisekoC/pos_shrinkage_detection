import pandas as pd
import numpy as np
from collections import defaultdict

# Read master data
employee_df = pd.read_csv('/content/drive/MyDrive/employee_master.csv')
store_df = pd.read_csv('/content/drive/MyDrive/store_master.csv')

# Aggregation dictionary: key = (employee_id, year_month)
emp_month = defaultdict(lambda: {
    'refund_count': 0,
    'void_count': 0,
    'override_count': 0,
    'transaction_count': 0,
    'cash_count': 0,
    'refund_no_receipt_count': 0,
    'high_discount_cash_count': 0,
    'sum_discount_pct': 0.0,
    'total_final_price': 0.0,
    'total_discount': 0.0,
    'late_night_count': 0,
    'high_risk_refund_count': 0
})

# Process transactions in chunks
chunk_size = 100000
for chunk in pd.read_csv('/content/drive/MyDrive/transactions.csv', chunksize=chunk_size, parse_dates=['transaction_timestamp']):
    chunk['year_month'] = chunk['transaction_timestamp'].dt.to_period('M')
    chunk['is_late_night'] = (chunk['transaction_timestamp'].dt.hour >= 20).astype(int)
    chunk['is_high_risk_item'] = (chunk['item_price'] > 100).astype(int)
    chunk['is_cash'] = (chunk['payment_type'] == 'Cash').astype(int)
    # Discount percent (avoid division by zero)
    chunk['discount_pct'] = chunk['discount_amount'] / (chunk['item_price'] * chunk['quantity']).clip(1)
    # Pattern flags
    chunk['high_discount_cash'] = ((chunk['discount_pct'] > 0.3) & (chunk['is_cash'] == 1)).astype(int)
    chunk['refund_no_receipt'] = ((chunk['refund_flag'] == 1) & (chunk['receipt_provided'] == 0)).astype(int)

    for _, row in chunk.iterrows():
        key = (row['employee_id'], row['year_month'])
        emp_month[key]['refund_count'] += row['refund_flag']
        emp_month[key]['void_count'] += row['void_flag']
        emp_month[key]['override_count'] += row['override_flag']
        emp_month[key]['transaction_count'] += 1
        emp_month[key]['cash_count'] += row['is_cash']
        emp_month[key]['refund_no_receipt_count'] += row['refund_no_receipt']
        emp_month[key]['high_discount_cash_count'] += row['high_discount_cash']
        emp_month[key]['sum_discount_pct'] += row['discount_pct']
        emp_month[key]['total_discount'] += row['discount_amount']
        emp_month[key]['total_final_price'] += row['final_price']
        emp_month[key]['late_night_count'] += row['is_late_night']
        if row['refund_flag'] and row['is_high_risk_item']:
            emp_month[key]['high_risk_refund_count'] += 1

# Convert to DataFrame
records = []
for (emp_id, ym), vals in emp_month.items():
    records.append({
        'employee_id': emp_id,
        'year_month': str(ym),
        **vals
    })
monthly_df = pd.DataFrame(records)

# Calculate rates
monthly_df['refund_rate'] = monthly_df['refund_count'] / monthly_df['transaction_count']
monthly_df['void_rate'] = monthly_df['void_count'] / monthly_df['transaction_count']
monthly_df['override_rate'] = monthly_df['override_count'] / monthly_df['transaction_count']
monthly_df['avg_discount_pct'] = monthly_df['sum_discount_pct'] / monthly_df['transaction_count']
monthly_df['cash_rate'] = monthly_df['cash_count'] / monthly_df['transaction_count']
monthly_df['no_receipt_refund_rate'] = monthly_df['refund_no_receipt_count'] / monthly_df['refund_count'].clip(1)
monthly_df['high_discount_cash_rate'] = monthly_df['high_discount_cash_count'] / monthly_df['transaction_count']  # optional
monthly_df['avg_transaction_value'] = monthly_df['total_final_price'] / monthly_df['transaction_count']
monthly_df['late_night_txn_pct'] = monthly_df['late_night_count'] / monthly_df['transaction_count']
monthly_df['transaction_frequency'] = monthly_df['transaction_count']
monthly_df['high_risk_sku_refund_ratio'] = monthly_df['high_risk_refund_count'] / monthly_df['refund_count'].clip(1)

# Fill NaN for cases where refund_count=0
monthly_df['no_receipt_refund_rate'] = monthly_df['no_receipt_refund_rate'].fillna(0)

# Sort for rolling
monthly_df = monthly_df.sort_values(['employee_id', 'year_month'])

# Rolling 90-day (3-month) sums (optional, you can keep or drop)
rolling_cols = [
    'refund_count', 'void_count', 'override_count', 'transaction_count',
    'total_discount', 'total_final_price', 'late_night_count', 'high_risk_refund_count',
    'cash_count', 'refund_no_receipt_count', 'sum_discount_pct', 'high_discount_cash_count'
]
for col in rolling_cols:
    monthly_df[f'rolling90_{col}'] = monthly_df.groupby('employee_id')[col].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )

# Compute rolling rates (if desired)
# ... (you can add if needed)

# Add employee and store info
monthly_df = monthly_df.merge(employee_df, on='employee_id')
monthly_df = monthly_df.merge(store_df, on='store_id')

# Save
monthly_df.to_csv('/content/drive/MyDrive/features_employee_monthly.csv', index=False)
print("Employee features saved with pattern counts.")