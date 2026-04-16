"""
Interface Streamlit — Fraud Detection v2.0
Thème : Rouge Banque (blanc + bordeaux + gris anthracite)
Version autonome pour Streamlit Cloud.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fraud Detector", page_icon="🏦", layout="centered")

@st.cache_resource
def load_artifact():
    # On garde la même interface, mais on charge le modèle localement
    # depuis artifacts/model.joblib à la racine du repo.
    base_dir = Path(__file__).resolve().parents[1]
    candidates = [
        base_dir / "artifacts" / "model.joblib",
        Path("artifacts/model.joblib"),
        Path("./artifacts/model.joblib"),
    ]

    for path in candidates:
        if path.exists():
            return joblib.load(path), path

    raise FileNotFoundError(
        "Impossible de trouver artifacts/model.joblib. "
        "Ajoute ton vrai modèle d'origine dans le dossier artifacts."
    )

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; background:#FFFFFF; color:#2D2D2D; }
.stApp { background-color:#F7F7F7; }
.header-bar { background:#8B1A2B; padding:1.4rem 2rem 1rem; margin:-1rem -1rem 1.5rem; border-bottom:4px solid #5C0F1A; }
.header-bar h1 { font-family:'Playfair Display',serif!important; color:white!important; font-size:2rem!important; margin:0!important; }
.header-bar p { color:rgba(255,255,255,0.75)!important; margin:0.3rem 0 0!important; font-size:0.85rem!important; }
.badge { display:inline-block; background:rgba(255,255,255,0.15); color:white; font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; padding:0.2rem 0.8rem; border-radius:999px; border:1px solid rgba(255,255,255,0.3); margin-bottom:0.5rem; }
.card { background:white; border:1px solid #E0E0E0; border-radius:10px; padding:1.2rem 1.5rem; margin-bottom:1rem; box-shadow:0 2px 8px rgba(0,0,0,0.06); }
.card-title { font-weight:600; font-size:0.82rem; letter-spacing:0.08em; text-transform:uppercase; color:#8B1A2B; margin-bottom:0.8rem; padding-bottom:0.5rem; border-bottom:2px solid #8B1A2B; }
.metric-row { display:flex; gap:0.8rem; margin-bottom:1rem; }
.metric-box { flex:1; background:white; border:1px solid #E0E0E0; border-top:3px solid #8B1A2B; border-radius:8px; padding:0.7rem; text-align:center; }
.metric-val { font-size:1.4rem; font-weight:700; color:#8B1A2B; }
.metric-lbl { font-size:0.72rem; color:#888; margin-top:0.1rem; }
.stButton > button { background:#8B1A2B!important; color:white!important; font-weight:600!important; border:none!important; border-radius:8px!important; padding:0.7rem 2rem!important; width:100%; font-family:'Inter',sans-serif!important; font-size:0.95rem!important; }
.stButton > button:hover { opacity:0.85!important; }
.result-card { background:white; border-radius:12px; padding:1.8rem 2rem; margin-top:1.2rem; text-align:center; box-shadow:0 4px 20px rgba(0,0,0,0.08); }
.result-card.fraud { border:2px solid #C0392B; border-top:5px solid #C0392B; }
.result-card.legit { border:2px solid #1A7A3C; border-top:5px solid #1A7A3C; }
.result-icon { font-size:2.5rem; display:block; margin-bottom:0.6rem; }
.result-label { font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:700; }
.result-label.fraud { color:#C0392B; }
.result-label.legit { color:#1A7A3C; }
.proba-track { background:#F0F0F0; border-radius:6px; height:10px; margin:0.8rem auto; max-width:320px; overflow:hidden; }
.proba-fill { height:10px; border-radius:6px; }
.result-meta { color:#888; font-size:0.8rem; margin-top:0.6rem; }
.status-ok  { color:#1A7A3C; font-weight:600; font-size:0.9rem; }
.status-err { color:#C0392B; font-weight:600; font-size:0.9rem; }
hr { border:none; border-top:1px solid #E8E8E8!important; margin:1.2rem 0!important; }
label { font-weight:500!important; color:#2D2D2D!important; font-size:0.88rem!important; }
.small-note { color:#777; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
  <div class="badge">TP MLOps v2.0 · Vrai modèle · 100k transactions</div>
  <h1>🏦 Fraud Detector</h1>
  <p>Détection de transactions frauduleuses — ROC-AUC 95% · Recall Fraude 87%</p>
</div>
""", unsafe_allow_html=True)

artifact = None
artifact_path = None
load_error = None

try:
    with st.spinner("Chargement du modèle..."):
        artifact, artifact_path = load_artifact()
except Exception as e:
    load_error = str(e)

st.markdown('<div class="card"><div class="card-title">État de l\'API</div>', unsafe_allow_html=True)

if load_error:
    st.markdown('<p class="status-err">● Erreur de chargement du modèle</p>', unsafe_allow_html=True)
    st.error(load_error)
    st.markdown('<p class="small-note">Vérifie que ton vrai modèle est bien dans <code>artifacts/model.joblib</code>.</p>', unsafe_allow_html=True)
else:
    version  = artifact.get("version", "—")
    roc_auc  = artifact.get("roc_auc", 0.0) * 100
    accuracy = artifact.get("accuracy", 0.0) * 100

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box"><div class="metric-val">✅</div><div class="metric-lbl">API en ligne</div></div>
      <div class="metric-box"><div class="metric-val">{version}</div><div class="metric-lbl">Version</div></div>
      <div class="metric-box"><div class="metric-val">{roc_auc:.1f}%</div><div class="metric-lbl">ROC-AUC</div></div>
      <div class="metric-box"><div class="metric-val">{accuracy:.1f}%</div><div class="metric-lbl">Accuracy</div></div>
    </div>
    <p class="status-ok">● Modèle chargé et opérationnel</p>
    """, unsafe_allow_html=True)
    st.caption(f"Chemin détecté : {artifact_path}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">Analyser une transaction</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    amount     = st.number_input("Montant (€)", min_value=0.01, value=90.0, step=5.0)
    age        = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
    hour       = st.slider("Heure de la transaction", 0, 23, 14)
    days_since = st.number_input("Jours depuis dernière transaction", min_value=0, value=7)

with c2:
    gender_lbl  = st.selectbox("Genre", ["Homme (M)", "Femme (F)"])
    gender      = "M" if "(M)" in gender_lbl else "F"
    category    = st.selectbox("Catégorie", [
        "grocery","food","travel","entertainment",
        "health","electronics","clothing","fuel","online","atm"
    ])
    foreign_lbl = st.selectbox("Transaction à l'étranger", ["Non (0)", "Oui (1)"])
    is_foreign  = 1 if "(1)" in foreign_lbl else 0
    st.markdown("**Payload envoyé à l'API :**")
    st.code(
        f'{{"amount": {amount}, "age": {age}, "transaction_hour": {hour}, '
        f'"days_since_last_txn": {days_since}, "gender": "{gender}", '
        f'"category": "{category}", "is_foreign_country": {is_foreign}}}',
        language="json"
    )

st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔎 Analyser la transaction"):
    if load_error or artifact is None:
        st.error("❌ Le modèle n'est pas chargé. Corrige d'abord artifacts/model.joblib.")
    else:
        row = pd.DataFrame([{
            "amount":              amount,
            "transaction_hour":    hour,
            "days_since_last_txn": days_since,
            "is_foreign_country":  is_foreign,
            "age":                 age,
            "gender":              gender,
            "category":            category,
        }])

        try:
            pred  = int(artifact["model"].predict(row)[0])
            proba = float(artifact["model"].predict_proba(row)[0][1])
            label = "Fraude" if pred == 1 else "Légitime"
            css   = "fraud" if pred == 1 else "legit"
            icon  = "🚨" if pred == 1 else "✅"
            bar   = "#C0392B" if pred == 1 else "#1A7A3C"

            st.markdown(f"""
            <div class="result-card {css}">
              <span class="result-icon">{icon}</span>
              <div class="result-label {css}">{label}</div>
              <div style="margin:0.4rem auto 0;max-width:320px;">
                <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-bottom:4px;">
                  <span>Légitime</span><span>Fraude</span>
                </div>
                <div class="proba-track">
                  <div class="proba-fill" style="width:{proba*100:.0f}%;background:{bar};"></div>
                </div>
                <div style="font-size:0.88rem;color:{bar};font-weight:600;margin-top:4px;">
                  Probabilité de fraude : {proba*100:.1f}%
                </div>
              </div>
              <div class="result-meta">Modèle {artifact.get('version','—')} · ROC-AUC {artifact.get('roc_auc',0)*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

st.markdown("---")
st.markdown('<p style="color:#AAA;font-size:0.75rem;text-align:center;">TP MLOps 2025-2026 · Wilfried MBEUMI · Eddie ATINDEHOU · Joël ONANA</p>', unsafe_allow_html=True)
