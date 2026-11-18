import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = r"C:\Users\KISHORE\Downloads\archive"
MODEL_DIR = r"E:\vaulto-ai\app\saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

files_to_load = [
    os.path.join(DATA_DIR, "budgetwise_finance_dataset.csv"),
    os.path.join(DATA_DIR, "budgetwise_synthetic_dirty.csv"),
]

print("📂 Ingesting training datasets...")
dfs = []
for file_path in files_to_load:
    if os.path.exists(file_path):
        print(f" - Loading: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()
        dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# Standardize columns
column_mapping = {
    'transaction_amount': 'amount',
    'expense_amount': 'amount',
    'trans_date': 'date',
    'transaction_date': 'date',
    'category_name': 'category',
    'notes': 'description'
}
combined_df.rename(columns=column_mapping, inplace=True)

# Clean numeric amounts and categories
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

# ---------------------------------------------------------
# 1. Train Isolation Forest Anomaly Model
# ---------------------------------------------------------
print("\n🤖 1/3 Training Isolation Forest Anomaly Model...")
X_iso = np.log1p(combined_df[["amount"]].values)
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_iso)
joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.joblib"))

# ---------------------------------------------------------
# 2. Compute Category Baselines
# ---------------------------------------------------------
print("\n📊 2/3 Computing Category Average Baselines...")
category_baselines = combined_df.groupby("category")["amount"].mean().to_dict()
joblib.dump(category_baselines, os.path.join(MODEL_DIR, "category_baselines.joblib"))

# ---------------------------------------------------------
# 3. Train TF-IDF + Random Forest Classifier
# ---------------------------------------------------------
print("\n🏷️ 3/3 Training TF-IDF + Random Forest Transaction Categorizer...")
# Filter valid text entries
text_df = combined_df[combined_df['description'].str.strip() != ''].copy()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, stop_words='english')
X_text = vectorizer.fit_transform(text_df['description'])
y_labels = text_df['category']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_text, y_labels)

joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(clf, os.path.join(MODEL_DIR, "category_classifier.joblib"))

print("\n🎉 Training Complete! Saved models to E:\\vaulto-ai\\app\\saved_models\\:")
print(" - isolation_forest.joblib")
print(" - category_baselines.joblib")
print(" - tfidf_vectorizer.joblib")
print(" - category_classifier.joblib")
