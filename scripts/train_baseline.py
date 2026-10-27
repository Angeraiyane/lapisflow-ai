import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import os

# 1. Chargement des jeux de données
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")

# 2. Pipeline scikit-learn calibré et léger
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
])

print("Entraînement du classifieur v0 baseline...")
pipeline.fit(train_df["text"], train_df["category"])

# 3. Évaluation sur l'ensemble de validation
preds = pipeline.predict(val_df["text"])
macro_f1 = f1_score(val_df["category"], preds, average="macro")

print("\n--- RAPPORT DE CLASSIFICATION BASELINE (VALIDATION) ---")
print(classification_report(val_df["category"], preds))
print(f"Validation Macro-F1: {macro_f1:.4f}")

# 4. Sauvegarde de l'artefact
os.makedirs("models", exist_ok=True)
model_path = "models/clf_v0.joblib"
joblib.dump(pipeline, model_path)
size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"\nModèle sauvegardé dans {model_path} ({size_mb:.2f} MB)")