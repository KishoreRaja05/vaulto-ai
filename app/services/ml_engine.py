import os
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from app.models.schemas import TransactionItem

# Model storage directory, relative to this file
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "saved_models"))

ISO_FOREST_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
ISO_SCORES_PATH = os.path.join(MODEL_DIR, "isolation_scores.joblib")
BASELINES_PATH = os.path.join(MODEL_DIR, "category_baselines.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "category_classifier.joblib")


def _load(path: str, default=None):
    return joblib.load(path) if os.path.exists(path) else default


pretrained_iso_forest = _load(ISO_FOREST_PATH)
pretrained_scores = _load(ISO_SCORES_PATH)
pretrained_baselines = _load(BASELINES_PATH, {})
pretrained_vectorizer = _load(VECTORIZER_PATH)
pretrained_classifier = _load(CLASSIFIER_PATH)


def predict_spending_forecast(transactions: List[TransactionItem]) -> Dict[str, Any]:
    """Project month-end spend with an honest fit statistic and interval."""
    expenses = [t for t in transactions if t.type == "expense" and t.amount > 0]

    if not expenses:
        return {
            "projected_month_end_spend": 0.0,
            "daily_avg_spend": 0.0,
            "model_r2": None,
            "prediction_interval_90": [0.0, 0.0],
            "insights": ["No expense transactions available to generate a forecast."],
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

    insights = [
        f"You have spent ₹{total_spent_so_far:,.2f} over the first {current_day} days of the month.",
        f"Your current daily spending velocity is ₹{daily_avg:,.2f}/day.",
    ]

    if len(daily_spend) >= 3:
        X = daily_spend[["day"]].values
        y = daily_spend["cumulative_spend"].values

        model = LinearRegression().fit(X, y)
        projected_spend = max(float(model.predict(np.array([[30]]))[0]), total_spent_so_far)
        model_r2 = float(model.score(X, y))
        residuals = y - model.predict(X)
        margin = 1.645 * float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
        low = max(projected_spend - margin, total_spent_so_far)
        high = projected_spend + margin
        insights.append(
            f"Linear trend R^2 on observed days: {model_r2:.2f}; "
            f"90% residual interval: ₹{low:,.2f} - ₹{high:,.2f}."
        )
    else:
        # Too little history to fit a trend: plain extrapolation, no R^2.
        projected_spend = daily_avg * 30.0
        model_r2 = None
        low, high = projected_spend * 0.75, projected_spend * 1.25
        insights.append(
            "Fewer than 3 spending days: simple extrapolation used, no regression fit. "
            f"Indicative range: ₹{low:,.2f} - ₹{high:,.2f}."
        )

    return {
        "projected_month_end_spend": round(projected_spend, 2),
        "daily_avg_spend": round(daily_avg, 2),
        "model_r2": round(model_r2, 3) if model_r2 is not None else None,
        "prediction_interval_90": [round(low, 2), round(high, 2)],
        "insights": insights,
    }


def detect_spending_anomalies(
    transactions: List[TransactionItem], contamination: float = 0.05
) -> Dict[str, Any]:
    """Flag outliers; the caller's contamination sets the score threshold."""
    expenses = [t for t in transactions if t.type == "expense" and t.amount > 0]

    if not expenses:
        return {"anomalies_found": 0, "anomalies": []}

    df = pd.DataFrame([t.model_dump() for t in expenses])
    X = np.log1p(df[["amount"]].values)
    contamination = float(np.clip(contamination, 0.001, 0.5))

    if pretrained_iso_forest is not None:
        scores = pretrained_iso_forest.decision_function(X)
        if pretrained_scores is not None:
            threshold = float(np.percentile(pretrained_scores, contamination * 100))
            labels = np.where(scores <= threshold, -1, 1)
        else:
            labels = pretrained_iso_forest.predict(X)
    else:
        fallback = IsolationForest(contamination=contamination, random_state=42)
        labels = fallback.fit_predict(X)
        scores = fallback.decision_function(X)

    df["anomaly_label"] = labels
    df["anomaly_score"] = scores

    anomalies_list = []
    for _, row in df[df["anomaly_label"] == -1].iterrows():
        category = str(row["category"])
        cat_avg = pretrained_baselines.get(
            category, df[df["category"] == category]["amount"].mean()
        )
        if pd.isna(cat_avg) or cat_avg == 0:
            cat_avg = df["amount"].mean()

        direction = "high" if row["amount"] > cat_avg else "low"
        anomalies_list.append({
            "transaction_id": row.get("id"),
            "amount": float(row["amount"]),
            "category": category,
            "date": str(row["date"]),
            "anomaly_score": round(float(row["anomaly_score"]), 4),
            "reason": (
                f"Expense of ₹{row['amount']:,.2f} is unusually {direction} for {category} "
                f"(category average ₹{cat_avg:,.2f})."
            ),
        })

    return {"anomalies_found": len(anomalies_list), "anomalies": anomalies_list}


def predict_transaction_category(description: str) -> Dict[str, Any]:
    if not pretrained_vectorizer or not pretrained_classifier:
        return {
            "description": description,
            "predicted_category": "Other",
            "confidence": 0.0,
            "message": "Categorizer model weights not loaded.",
        }

    X_text = pretrained_vectorizer.transform([description])
    predicted_category = str(pretrained_classifier.predict(X_text)[0])
    probabilities = pretrained_classifier.predict_proba(X_text)
    confidence = float(np.max(probabilities))

    return {
        "description": description,
        "predicted_category": predicted_category,
        "confidence": round(confidence, 2),
    }
