"""
Schémas Pydantic — validation des entrées/sorties
"""
from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    amount:               float = Field(..., gt=0,   description="Montant de la transaction en € (> 0)")
    transaction_hour:     int   = Field(..., ge=0, le=23, description="Heure de la transaction (0-23)")
    days_since_last_txn:  int   = Field(..., ge=0,   description="Jours depuis la dernière transaction")
    is_foreign_country:   int   = Field(..., ge=0, le=1,  description="Transaction à l'étranger : 0=Non, 1=Oui")
    age:                  int   = Field(..., ge=18, le=100, description="Âge du client")
    gender:               Literal["M", "F"] = Field(..., description="Genre : M ou F")
    category:             Literal[
                              "grocery","food","travel","entertainment",
                              "health","electronics","clothing","fuel","online","atm"
                          ] = Field(..., description="Catégorie de la transaction")

class PredictResponse(BaseModel):
    prediction:     int
    label:          str
    probability:    float
    model_version:  str
    roc_auc:        float
