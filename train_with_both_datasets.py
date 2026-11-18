import pandas as pd
from app.models.schemas import TransactionItem
from app.services.ml_engine import predict_spending_forecast, detect_spending_anomalies

# ---------------------------------------------------------
# Step 1: Load Both Datasets
# ---------------------------------------------------------
# Replace with your actual file paths (e.g., dataset1.csv, dataset2.csv)
df1 = pd.read_csv("data/budgetwise_dataset_1.csv")
df2 = pd.read_csv("data/budgetwise_dataset_2.csv")

# ---------------------------------------------------------
# Step 2: Standardize & Concatenate (Merge)
# ---------------------------------------------------------
# Ensure required columns exist: ['amount', 'category', 'date', 'type']
combined_df = pd.concat([df1, df2], ignore_index=True)

# Clean date formatting & remove duplicates if any
combined_df["date"] = pd.to_datetime(combined_df["date"]).dt.strftime("%Y-%m-%d")
combined_df = combined_df.drop_duplicates().sort_values("date")

print(f"✅ Combined Total Transactions: {len(combined_df)} rows")

# ---------------------------------------------------------
# Step 3: Convert to Pydantic Schemas
# ---------------------------------------------------------
transactions = [
    TransactionItem(
        amount=float(row["amount"]),
        category=str(row["category"]),
        date=str(row["date"]),
        type=str(row.get("type", "expense")),
        description=str(row.get("description", ""))
    )
    for _, row in combined_df.iterrows()
]

# ---------------------------------------------------------
# Step 4: Run Machine Learning Models on Combined Data
# ---------------------------------------------------------
print("\n--- 📈 Spending Forecast Result ---")
forecast_result = predict_spending_forecast(transactions)
print(f"Projected Month-End Total: ₹{forecast_result['projected_month_end_spend']:,.2f}")
print(f"Daily Spending Velocity:   ₹{forecast_result['daily_avg_spend']:,.2f}/day")
print(f"Model Confidence Score:   {forecast_result['confidence_score'] * 100}%")
for insight in forecast_result["insights"]:
    print(f" - {insight}")

print("\n--- 🚨 Anomaly Detection Result ---")
anomaly_result = detect_spending_anomalies(transactions, contamination=0.05)
print(f"Anomalies Found: {anomaly_result['anomalies_found']}")
for anomaly in anomaly_result["anomalies"]:
    print(f" ⚠️ [{anomaly['date']}] {anomaly['category']}: ₹{anomaly['amount']:,.2f} -> {anomaly['reason']}")