# Fraud Detector - MLOps FastAPI & Streamlit
ROC-AUC 95% - Recall Fraude 87% - 100k transactions

## Lancement
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.api:app --reload
streamlit run ui/streamlit_app.py

## Auteurs
Wilfried MBEUMI - Eddie ATINDEHOU - Joel ONANA
TP MLOps 2025-2026
