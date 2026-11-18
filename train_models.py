"""Reproducible training for the Vaulto AI ML artifacts.

Run:  python train_models.py
Reads only data/*.csv next to this file and writes app/saved_models/*.joblib.
"""

import difflib
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "app" / "saved_models"
DATA_FILES = ("budgetwise_finance_dataset.csv", "budgetwise_synthetic_dirty.csv")

CANONICAL = [
    "Food", "Transport", "Shopping", "Bills", "Entertainment",
    "Healthcare", "Education", "Rent", "Salary", "Freelance", "Other",
]

# Canonical names and common spellings used as fuzzy-match targets.
LOOKUP = {name.lower(): name for name in CANONICAL}
LOOKUP.update({
    # Food
    "fod": "Food", "foods": "Food", "foodd": "Food", "foood": "Food",
    "ffood": "Food", "fodo": "Food", "ofod": "Food", "ood": "Food", "foo": "Food",
    # Rent
    "rentt": "Rent", "rnt": "Rent", "reent": "Rent", "ret": "Rent",
    "retn": "Rent", "ernt": "Rent", "rnet": "Rent", "rennt": "Rent",
    # Transport
    "travel": "Transport", "traval": "Transport", "travl": "Transport",
    "ravel": "Transport", "tarvel": "Transport", "rtavel": "Transport",
    "travell": "Transport", "trvael": "Transport", "travle": "Transport",
    # Bills / utilities
    "utility": "Bills", "utilities": "Bills", "utilties": "Bills",
    "utlities": "Bills", "uitlities": "Bills", "utillities": "Bills",
    "utilitie": "Bills", "utilise": "Bills",
    # Healthcare
    "health": "Healthcare", "helth": "Healthcare",
    # Entertainment
    "entertain": "Entertainment", "entertainment": "Entertainment",
    "ent": "Entertainment",
    # Education
    "edu": "Education", "educaton": "Education", "edcation": "Education",
    "edducation": "Education",
    # Salary / freelance
    "salaryy": "Salary", "saary": "Salary", "saalry": "Salary",
    "salay": "Salary", "salry": "Salary", "slary": "Salary",
    "bonus": "Salary", "bbonus": "Salary", "bnous": "Salary",
    "bonnus": "Salary", "bons": "Salary", "obnus": "Salary", "onus": "Salary",
    "freelancing": "Freelance",
    # Other income and assorted noise
    "savings": "Other", "saving": "Other", "saings": "Other",
    "saivngs": "Other", "saviings": "Other", "savinggs": "Other",
    "svings": "Other", "investment": "Other", "misc": "Other",
    "others": "Other", "other income": "Other", "oother income": "Other",
    "otehrs": "Other", "oother": "Other",
})

# The synthetic source assigns categories at random relative to the note text,
# so unambiguous descriptions are relabelled before training.
KEYWORD_RULES = [
    ("Food", ("dinner", "lunch", "breakfast", "restaurant", "resto", "coffee",
              "grocery", "groceries", "pizza", "food", "cafe", "snack")),
    ("Rent", ("rent", "landlord", "lease")),
    ("Transport", ("uber", "cab", "taxi", "metro", "bus", "fuel", "petrol",
                   "train", "flight", "travel", "commute")),
    ("Bills", ("electricity", "water bill", "gas bill", "internet", "utility", "bill")),
    ("Entertainment", ("movie", "netflix", "concert", "streaming", "cinema", "game")),
    ("Healthcare", ("doctor", "hospital", "pharmacy", "medicine", "clinic", "gym")),
    ("Education", ("course", "tuition", "school", "college", "book", "fee", "class")),
    ("Salary", ("salary", "paycheck", "payroll", "bonus")),
    ("Freelance", ("freelance", "client", "invoice")),
    ("Shopping", ("shopping", "amazon", "clothes", "shoes", "electronics")),
    ("Other", ("atm withdrawal", "gift", "misc", "investment", "saving")),
]


def canonicalize_category(raw_label: str) -> str:
    label = re.sub(r"[^a-z ]", " ", str(raw_label).lower().strip())
    label = re.sub(r"\s+", " ", label).strip()
    if label in LOOKUP:
        return LOOKUP[label]
    match = difflib.get_close_matches(label, LOOKUP.keys(), n=1, cutoff=0.65)
    return LOOKUP[match[0]] if match else "Other"


def category_from_description(description: str):
    for category, keywords in KEYWORD_RULES:
        if any(word in description for word in keywords):
            return category
    return None


def load_and_clean() -> pd.DataFrame:
    frames = []
    for name in DATA_FILES:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Missing training data: {path}")
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.lower().str.strip()
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["description"] = (
        df["notes"].fillna("").astype(str).str.lower().str.strip()
        if "notes" in df.columns else ""
    )
    df["amount"] = pd.to_numeric(
        df["amount"].astype(str).str.replace(r"[$,\s]", "", regex=True),
        errors="coerce",
    )
    df.dropna(subset=["amount"], inplace=True)
    df = df[df["amount"] > 0].copy()

    source_categories = df["category"].map(canonicalize_category)
    text_categories = df["description"].map(category_from_description)
    df["category"] = text_categories.fillna(source_categories)
    corrections = int((text_categories.notna() & (text_categories != source_categories)).sum())

    low = df.groupby("category")["amount"].transform(lambda s: s.quantile(0.01))
    high = df.groupby("category")["amount"].transform(lambda s: s.quantile(0.99))
    df["amount"] = df["amount"].clip(low, high)
    df = df[df["description"].str.len() > 0].copy()

    print(f"Loaded {len(df)} usable rows from {len(DATA_FILES)} files")
    print(f"Relabelled {corrections} rows from unambiguous description keywords")
    print("Category counts:", df["category"].value_counts().to_dict())
    return df


def train_categorizer(df: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        df["description"], df["category"], test_size=0.2,
        random_state=42, stratify=df["category"],
    )

    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), max_features=2000,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 max_features=2000, sublinear_tf=True)),
    ])
    X_train_vec = features.fit_transform(X_train)
    X_test_vec = features.transform(X_test)
    print(f"TF-IDF feature count: {X_train_vec.shape[1]}")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=5.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=20,
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        acc = accuracy_score(y_test, model.predict(X_test_vec))
        results[name] = acc
        print(f"{name} accuracy: {acc:.4f}")

    best_name = max(results, key=results.get)
    best = models[best_name]
    print(f"Selected: {best_name}")
    print(classification_report(y_test, best.predict(X_test_vec), zero_division=0))

    joblib.dump(features, MODEL_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(best, MODEL_DIR / "category_classifier.joblib")
    size_mb = (MODEL_DIR / "category_classifier.joblib").stat().st_size / 1_000_000
    print(f"Categorizer saved ({size_mb:.2f} MB)")

    sample = features.transform(["dinner at restaurant"])
    print(f"Check - 'dinner at restaurant' -> {best.predict(sample)[0]}")


def train_anomaly_model(df: pd.DataFrame) -> None:
    X = np.log1p(df[["amount"]].values)
    forest = IsolationForest(contamination=0.05, random_state=42).fit(X)
    joblib.dump(forest, MODEL_DIR / "isolation_forest.joblib")
    # Calibration scores let inference honour a caller's contamination value.
    joblib.dump(forest.decision_function(X), MODEL_DIR / "isolation_scores.joblib")

    baselines = df.groupby("category")["amount"].mean().to_dict()
    joblib.dump(baselines, MODEL_DIR / "category_baselines.joblib")
    print("Anomaly model and baselines saved")
    for category, avg in sorted(baselines.items(), key=lambda item: -item[1])[:5]:
        print(f"  {category}: {avg:,.2f}")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean()
    train_categorizer(df)
    train_anomaly_model(df)
    print(f"Artifacts written to {MODEL_DIR}")


if __name__ == "__main__":
    main()
