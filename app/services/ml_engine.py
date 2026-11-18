import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from app.models.schemas import TransactionItem

# Model storage directory
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_models"))

ISO_FOREST_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
BASELINES_PATH = os.path.join(MODEL_DIR, "category_baselines.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "category_classifier.joblib")


# =========================================================
# 1. MODEL TRAINING LOGIC
# =========================================================
def train_and_save_models(data_dir: str = r"C:\Users\KISHORE\Downloads\archive"):
    """
    Ingests raw datasets, trains Isolation Forest & Random Forest Categorizer,
    computes category baselines, and serializes .joblib files to disk.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    files_to_load = [
        os.path.join(data_dir, "budgetwise_finance_dataset.csv"),
        os.path.join(data_dir, "budgetwise_synthetic_dirty.csv"),
    ]

    print("📂 Ingesting training datasets...")
    dfs = []
    for file_path in files_to_load:
        if os.path.exists(file_path):
            print(f" - Loading: {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()
            dfs.append(df)

    if not dfs:
        print("❌ No valid training dataset files found.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    column_mapping = {
        'transaction_amount': 'amount',
        'expense_amount': 'amount',
        'trans_date': 'date',
        'transaction_date': 'date',
        'category_name': 'category',
        'notes': 'description'
    }
    combined_df.rename(columns=column_mapping, inplace=True)

    combined_df['amount'] = combined_df['amount'].astype(str).str.replace(r'[$,\s]', '', regex=True)
    combined_df['amount'] = pd.to_numeric(combined_df['amount'], errors='coerce')
    combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    combined_df.dropna(subset=['amount', 'date'], inplace=True)
    combined_df = combined_df[combined_df['amount'] > 0]

    if 'category' not in combined_df.columns:
        combined_df['category'] = 'General'
    combined_df['category'] = combined_df['category'].fillna('General').astype(str).str.strip().str.title()

    if 'description' not in combined_df.columns:
        combined_df['description'] = combined_df['category']
    combined_df['description'] = combined_df['description'].fillna(combined_df['category']).astype(str)

    print(f"✅ Preprocessed {len(combined_df)} records for training.")

    # 1. Train Isolation Forest
    print("\n🤖 Training Isolation Forest Anomaly Detection Model...")
    X_iso = np.log1p(combined_df[["amount"]].values)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    iso_forest.fit(X_iso)
    joblib.dump(iso_forest, ISO_FOREST_PATH)

    # 2. Compute Category Baselines
    print("📊 Computing Category Average Baselines...")
    category_baselines = combined_df.groupby("category")["amount"].mean().to_dict()
    joblib.dump(category_baselines, BASELINES_PATH)

    # 3. Train TF-IDF + Random Forest Classifier
    print("🏷️ Training TF-IDF + Random Forest Transaction Categorizer...")
    text_df = combined_df[combined_df['description'].str.strip() != ''].copy()

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, stop_words='english')
    X_text = vectorizer.fit_transform(text_df['description'])
    y_labels = text_df['category']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_text, y_labels)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, CLASSIFIER_PATH)

    print(f"\n🎉 Training Complete! Saved models to: {MODEL_DIR}")


# =========================================================
# 2. MODEL WEIGHTS INITIALIZATION
# =========================================================
def load_trained_models():
    iso_model = joblib.load(ISO_FOREST_PATH) if os.path.exists(ISO_FOREST_PATH) else None
    baselines = joblib.load(BASELINES_PATH) if os.path.exists(BASELINES_PATH) else {}
    vectorizer = joblib.load(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None
    classifier = joblib.load(CLASSIFIER_PATH) if os.path.exists(CLASSIFIER_PATH) else None
    return iso_model, baselines, vectorizer, classifier

pretrained_iso_forest, pretrained_baselines, pretrained_vectorizer, pretrained_classifier = load_trained_models()


# =========================================================
# 3. PREDICTION & INFERENCE FUNCTIONS
# =========================================================
def predict_spending_forecast(transactions: List[TransactionItem]) -> Dict[str, Any]:
    expenses = [t for t in transactions if t.type == "expense" and t.amount > 0]

    if not expenses:
        return {
            "projected_month_end_spend": 0.0,
            "daily_avg_spend": 0.0,
            "confidence_score": 0.0,
            "insights": ["No expense transactions available to generate a forecast."]
        }

    df = pd.DataFrame([t.model_dump() for t in expenses])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    daily_spend = df.groupby(df["date"].dt.day)["amount"].sum().reset_index()
    daily_spend.columns = ["day", "amount"]
    daily_spend["cumulative_spend"] = daily_spend["amount"].cumsum()

    current_day = int(daily_spend["day"].max())
    total_spent_so_far = float(daily_spend["cumulative_spend"].max())
    daily_avg = total_spent_so_far / max(current_day, 1)

    if len(daily_spend) >= 3:
        X = daily_spend[["day"]].values
        y = daily_spend["cumulative_spend"].values

        model = LinearRegression()
        model.fit(X, y)

        projected_spend = float(model.predict(np.array([[30]]))[0])
        projected_spend = max(projected_spend, total_spent_so_far)
        r2 = model.score(X, y)
        confidence = float(np.clip(r2, 0.50, 0.95))
    else:
        projected_spend = daily_avg * 30.0
        confidence = 0.60

    insights = [
        f"You have spent ₹{total_spent_so_far:,.2f} over the first {current_day} days of the month.",
        f"Your current daily spending velocity is ₹{daily_avg:,.2f}/day.",
        f"Projected month-end total expense: ₹{projected_spend:,.2f}."
    ]

    return {
        "projected_month_end_spend": round(projected_spend, 2),
        "daily_avg_spend": round(daily_avg, 2),
        "confidence_score": round(confidence, 2),
        "insights": insights
    }


def detect_spending_anomalies(transactions: List[TransactionItem], contamination: float = 0.05) -> Dict[str, Any]:
    expenses = [t for t in transactions if t.type == "expense" and t.amount > 0]

    if not expenses:
        return {"anomalies_found": 0, "anomalies": []}

    df = pd.DataFrame([t.model_dump() for t in expenses])
    X = np.log1p(df[["amount"]].values)

    if pretrained_iso_forest is not None:
        df["anomaly_label"] = pretrained_iso_forest.predict(X)
        df["anomaly_score"] = pretrained_iso_forest.decision_function(X)
    else:
        clf = IsolationForest(contamination=contamination, random_state=42)
        df["anomaly_label"] = clf.fit_predict(X)
        df["anomaly_score"] = clf.decision_function(X)

    anomalies_df = df[df["anomaly_label"] == -1]
    anomalies_list = []

    for _, row in anomalies_df.iterrows():
        category = str(row["category"])
        cat_avg = pretrained_baselines.get(category, df[df["category"] == category]["amount"].mean())
        if pd.isna(cat_avg) or cat_avg == 0:
            cat_avg = df["amount"].mean()

        reason = f"Expense of ₹{row['amount']:,.2f} is significantly higher than pre-trained category average (₹{cat_avg:,.2f})."

        anomalies_list.append({
            "transaction_id": row.get("id"),
            "amount": float(row["amount"]),
            "category": category,
            "date": str(row["date"]),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "reason": reason
        })

    return {
        "anomalies_found": len(anomalies_list),
        "anomalies": anomalies_list
    }


def predict_transaction_category(description: str) -> Dict[str, Any]:
    if not pretrained_vectorizer or not pretrained_classifier:
        return {
            "description": description,
            "predicted_category": "General",
            "confidence": 0.50,
            "message": "Categorizer model weights not loaded."
        }

    X_text = pretrained_vectorizer.transform([description])
    predicted_category = str(pretrained_classifier.predict(X_text)[0])
    probabilities = pretrained_classifier.predict_proba(X_text)
    confidence = float(np.max(probabilities))

    return {
        "description": description,
        "predicted_category": predicted_category,
        "confidence": round(confidence, 2)
    }


if __name__ == "__main__":
    train_and_save_models()
