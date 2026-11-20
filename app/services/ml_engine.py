import calendar
import os
from datetime import date
from typing import Any, Dict, List, Optional

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


MIN_TRANSACTIONS_FOR_ML = 30
MIN_FULL_MONTHS_FOR_ML = 2


def _as_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _in_month(item: TransactionItem, year: int, month: int) -> bool:
    parsed = _as_date(item.date)
    return bool(parsed and parsed.year == year and parsed.month == month)


def _split_one_offs(month_expenses: List[TransactionItem], all_expenses: List[TransactionItem]):
    """Separate one-off expenses from the regular ones used for the daily rate."""
    if not month_expenses:
        return [], [], 0.0

    amounts = np.array([float(t.amount) for t in all_expenses], dtype=float)
    threshold = float(np.mean(amounts)) + 2 * float(np.std(amounts))
    month_total = sum(float(t.amount) for t in month_expenses)

    regular: List[TransactionItem] = []
    excluded: List[TransactionItem] = []
    details: List[Dict[str, Any]] = []
    for item in month_expenses:
        amount = float(item.amount)
        reasons = []
        if amount > threshold:
            reasons.append("above mean + 2 standard deviations of your expenses")
        if month_total > 0 and amount / month_total > 0.40:
            reasons.append("more than 40% of this month's expenses")
        if reasons:
            excluded.append(item)
            details.append({
                "transaction_id": item.id,
                "amount": round(amount, 2),
                "category": item.category,
                "date": str(item.date),
                "reason": "Excluded as a one-off: " + " and ".join(reasons) + ".",
            })
        else:
            regular.append(item)

    # Never throw away every data point: with a single huge transaction there is
    # no regular rate to extrapolate, so treat it as regular instead.
    if not regular:
        return list(month_expenses), [], 0.0

    return regular, details, sum(float(t.amount) for t in excluded)


def predict_spending_forecast(
    transactions: List[TransactionItem], budget: Optional[float] = None
) -> Dict[str, Any]:
    """Project month-end spend with a structured context for the chat LLM.

    The trained model is only used with at least 30 transactions and 2 complete
    months of history. Otherwise a rule-based projection is used, one-off
    expenses are not extrapolated, and no confidence interval is produced.
    """
    parsed_dates = [parsed for parsed in (_as_date(t.date) for t in transactions) if parsed]
    if parsed_dates:
        reference = max(parsed_dates)
        year, month = reference.year, reference.month
        distinct_months = {(d.year, d.month) for d in parsed_dates}
        full_months = sum(1 for ym in distinct_months if ym < (year, month))
    else:
        today = date.today()
        year, month = today.year, today.month
        full_months = 0

    days_in_month = calendar.monthrange(year, month)[1]
    today = date.today()
    if (year, month) == (today.year, today.month):
        days_elapsed = min(today.day, days_in_month)
    else:
        days_elapsed = days_in_month
    days_remaining = max(days_in_month - days_elapsed, 0)

    month_items = [t for t in transactions if _in_month(t, year, month)]
    all_expenses = [t for t in transactions if t.type == "expense" and t.amount > 0]
    month_expenses = [t for t in month_items if t.type == "expense" and t.amount > 0]
    month_income = sum(float(t.amount) for t in month_items if t.type == "income" and t.amount > 0)
    total_spent = sum(float(t.amount) for t in month_expenses)
    daily_avg = total_spent / max(days_elapsed, 1)

    by_category: Dict[str, float] = {}
    for item in month_expenses:
        by_category[item.category] = by_category.get(item.category, 0.0) + float(item.amount)
    top_category, top_category_amount = max(by_category.items(), key=lambda kv: kv[1]) if by_category else (None, 0.0)

    enough_data = (
        len(transactions) >= MIN_TRANSACTIONS_FOR_ML
        and full_months >= MIN_FULL_MONTHS_FOR_ML
    )
    use_ml = enough_data and bool(month_expenses)

    excluded_details: List[Dict[str, Any]] = []
    model_r2: Optional[float] = None
    prediction_interval: Optional[List[float]] = None

    if not month_expenses:
        projected_spend = 0.0
        daily_avg = 0.0
        confidence = "low"
        insights = ["No expense transactions available to generate a forecast."]
    elif not use_ml:
        regular, excluded_details, one_off_total = _split_one_offs(month_expenses, all_expenses)
        regular_total = total_spent - one_off_total
        daily_rate = regular_total / max(days_elapsed, 1)
        projected_spend = daily_rate * days_in_month + one_off_total
        confidence = "low"
        insights = [
            "This is a rough estimate because there is little history.",
            f"Regular spending is about ₹{daily_rate:,.0f} per day over {days_elapsed} days.",
        ]
        if excluded_details:
            insights.append(
                f"{len(excluded_details)} one-off expense(s) totalling ₹{one_off_total:,.0f} "
                "were added back as a fixed amount instead of being extrapolated."
            )
    else:
        df = pd.DataFrame([t.model_dump() for t in month_expenses])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        daily_spend = df.groupby(df["date"].dt.day)["amount"].sum().reset_index()
        daily_spend.columns = ["day", "amount"]
        daily_spend["cumulative_spend"] = daily_spend["amount"].cumsum()

        if len(daily_spend) >= 3:
            X = daily_spend[["day"]].values
            y = daily_spend["cumulative_spend"].values
            model = LinearRegression().fit(X, y)
            projected_spend = max(float(model.predict(np.array([[days_in_month]]))[0]), total_spent)
            model_r2 = float(model.score(X, y))
            residuals = y - model.predict(X)
            margin = 1.645 * float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
            low = max(projected_spend - margin, total_spent)
            high = projected_spend + margin
            prediction_interval = [round(low, 2), round(high, 2)]
            insights = [
                f"You have spent ₹{total_spent:,.2f} over the first {days_elapsed} days of the month.",
                f"Linear trend R^2 on observed days: {model_r2:.2f}; "
                f"90% residual interval: ₹{low:,.2f} - ₹{high:,.2f}.",
            ]
        else:
            projected_spend = daily_avg * days_in_month
            low, high = projected_spend * 0.75, projected_spend * 1.25
            prediction_interval = [round(low, 2), round(high, 2)]
            insights = [
                "Fewer than 3 spending days: simple extrapolation used, no regression fit. "
                f"Indicative range: ₹{low:,.2f} - ₹{high:,.2f}."
            ]
        confidence = (
            "high" if full_months >= 6 and model_r2 is not None and model_r2 >= 0.7 else "medium"
        )

    context = {
        "month_income": round(month_income, 2),
        "month_expenses_so_far": round(total_spent, 2),
        "days_elapsed": days_elapsed,
        "days_in_month": days_in_month,
        "days_remaining": days_remaining,
        "projected_month_end_expenses": round(projected_spend, 2),
        "budget": budget,
        "top_category": top_category,
        "top_category_amount": round(float(top_category_amount), 2),
        "number_of_transactions": len(transactions),
        "months_of_history": full_months,
        "confidence": confidence,
        "excluded_outliers": excluded_details,
    }

    return {
        "projected_month_end_spend": round(projected_spend, 2),
        "daily_avg_spend": round(daily_avg, 2),
        "model_r2": round(model_r2, 3) if model_r2 is not None else None,
        "prediction_interval_90": prediction_interval,
        "confidence": confidence,
        "forecast_method": "ml" if use_ml else "rule_based",
        "insights": insights,
        "context": context,
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
