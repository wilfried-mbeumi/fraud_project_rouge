"""
API FastAPI — routes /health et /predict
"""
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import PredictRequest, PredictResponse
from app.model  import load_artifact

app = FastAPI(
    title="Fraud Detection API",
    description="Détection de fraude bancaire — TP MLOps v2.0",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Chargement unique au démarrage
artifact = load_artifact()

@app.get("/health", tags=["Monitoring"])
def health():
    return {
        "status":       "ok",
        "version":      artifact["version"],
        "features_num": artifact["features_num"],
        "features_cat": artifact["features_cat"],
        "accuracy":     artifact["accuracy"],
        "roc_auc":      artifact["roc_auc"],
    }

@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"])
def predict(payload: PredictRequest):
    try:
        row = pd.DataFrame([{
            "amount":              payload.amount,
            "transaction_hour":    payload.transaction_hour,
            "days_since_last_txn": payload.days_since_last_txn,
            "is_foreign_country":  payload.is_foreign_country,
            "age":                 payload.age,
            "gender":              payload.gender,
            "category":            payload.category,
        }])

        pred    = int(artifact["model"].predict(row)[0])
        proba   = float(artifact["model"].predict_proba(row)[0][1])
        label   = "Fraude" if pred == 1 else "Légitime"

        return PredictResponse(
            prediction=pred,
            label=label,
            probability=round(proba, 4),
            model_version=artifact["version"],
            roc_auc=artifact["roc_auc"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
