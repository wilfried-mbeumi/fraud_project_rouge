"""
Entraînement du modèle de détection de fraude
Dataset : 100 000 transactions réalistes — 1.5% de fraudes
"""
import joblib, os, numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ── Chargement ────────────────────────────────────────────────────────────────
df = pd.read_csv("data/transactions.csv")
print(f"Dataset chargé : {len(df):,} transactions — {df.fraud.mean()*100:.1f}% fraudes")

FEATURES_NUM = ["amount", "transaction_hour", "days_since_last_txn", "is_foreign_country", "age"]
FEATURES_CAT = ["gender", "category"]
FEATURES_ALL = FEATURES_NUM + FEATURES_CAT
TARGET       = "fraud"

X = df[FEATURES_ALL]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Pipeline ──────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES_CAT),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",  # gestion du déséquilibre 1.5% fraudes
        C=0.5
    ))
])

model.fit(X_train, y_train)

# ── Évaluation ────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc     = (y_pred == y_test).mean()
roc_auc = roc_auc_score(y_test, y_proba)
cm      = confusion_matrix(y_test, y_pred)

print("\n" + "="*55)
print(f"  Accuracy  : {acc:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}  ← métrique clé en détection de fraude")
print("="*55)

print("\n── Matrice de confusion ──────────────────────────────")
print(f"                   Prédit Légitime   Prédit Fraude")
print(f"  Réel Légitime        {cm[0][0]:>6}          {cm[0][1]:>5}")
print(f"  Réel Fraude          {cm[1][0]:>6}          {cm[1][1]:>5}")

print("\n── Rapport de classification ─────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Légitime", "Fraude"]))

# ── Sauvegarde ────────────────────────────────────────────────────────────────
artifact = {
    "model":        model,
    "features_num": FEATURES_NUM,
    "features_cat": FEATURES_CAT,
    "features_all": FEATURES_ALL,
    "version":      "v2.0",
    "accuracy":     round(acc, 4),
    "roc_auc":      round(roc_auc, 4),
}
os.makedirs("artifacts", exist_ok=True)
joblib.dump(artifact, "artifacts/model.joblib")
print("\n✅ Artefact sauvegardé : artifacts/model.joblib")
