import os
import pandas as pd
from app.models.schemas import TransactionItem
from app.services.ml_engine import predict_spending_forecast, detect_spending_anomalies

DATA_DIR = r"C:\Users\KISHORE\Downloads\archive"
files_to_load = [
    os.path.join(DATA_DIR, "budgetwise_finance_dataset.csv"),
    os.path.join(DATA_DIR, "budgetwise_synthetic_dirty.csv"),
]

print("📂 Loading selected datasets:")
dfs = []
for file_path in files_to_load:
    if os.path.exists(file_path):
        print(f" - {os.path.basename(file_path)}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()
        dfs.append(df)
    else:
        print(f"⚠️ File not found: {file_path}")

if not dfs:
    print("❌ No valid files loaded.")
    exit()

combined_df = pd.concat(dfs, ignore_index=True)
print(f"\n📊 Total raw records loaded: {len(combined_df)} rows")

column_mapping = {
    'transaction_amount': 'amount',
    'expense_amount': 'amount',
    'trans_date': 'date',
    'transaction_date': 'date',
    'category_name': 'category'
}
combined_df.rename(columns=column_mapping, inplace=True)

if 'amount' not in combined_df.columns:
    raise ValueError("Could not find an 'amount' column in the datasets.")

if 'date' not in combined_df.columns:
    combined_df['date'] = "2026-09-01"

if 'category' not in combined_df.columns:
    combined_df['category'] = "General"

if 'type' not in combined_df.columns:
    combined_df['type'] = "expense"

# Clean amount column: strip $, commas, and whitespace, then convert to numeric float
combined_df['amount'] = (
    combined_df['amount']
    .astype(str)
    .str.replace(r'[$,\s]', '', regex=True)
)
combined_df['amount'] = pd.to_numeric(combined_df['amount'], errors='coerce')

# Clean date column
combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

# Drop missing or non-positive amount/date rows
combined_df.dropna(subset=['amount', 'date'], inplace=True)
combined_df = combined_df[combined_df['amount'] > 0]

print(f"✅ Clean valid records after parsing: {len(combined_df)} rows")

transactions = [
    TransactionItem(
        amount=float(row['amount']),
        category=str(row['category']) if pd.notna(row['category']) else "General",
        date=str(row['date']),
        type=str(row['type']) if pd.notna(row['type']) else "expense",
        description=str(row.get('description', 'Budgetwise Data'))
    )
    for _, row in combined_df.iterrows()
]

print("\n==========================================")
print("📈 RUNNING SPENDING FORECAST MODEL")
print("==========================================")
forecast_result = predict_spending_forecast(transactions)
print(f"• Projected Month-End Spend: ₹{forecast_result['projected_month_end_spend']:,.2f}")
print(f"• Daily Spending Velocity:   ₹{forecast_result['daily_avg_spend']:,.2f}/day")
print(f"• Model Confidence Score:   {forecast_result['confidence_score'] * 100:.1f}%")
print("• Insights:")
for insight in forecast_result['insights']:
    print(f"   - {insight}")

print("\n==========================================")
print("🚨 RUNNING ISOLATION FOREST ANOMALY MODEL")
print("==========================================")
anomaly_result = detect_spending_anomalies(transactions, contamination=0.05)
print(f"• Total Anomalies Flagged: {anomaly_result['anomalies_found']}")
for idx, anomaly in enumerate(anomaly_result['anomalies'][:10], 1):
    print(f"   {idx}. [{anomaly['date']}] {anomaly['category']}: ₹{anomaly['amount']:,.2f} -> {anomaly['reason']}")
