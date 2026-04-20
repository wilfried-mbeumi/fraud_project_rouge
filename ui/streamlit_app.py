"""
Interface Streamlit — Fraud Detection v2.0
Thème : Rouge Banque (blanc + bordeaux + gris anthracite)
"""
import streamlit as st
import requests

# ⚠️ Remplace par ton URL Render si elle a changé
API_URL = "https://fraud-project-rouge.onrender.com"

st.set_page_config(page_title="Fraud Detector", page_icon="🏦", layout="centered")

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500;600&display=swap');

section[data-testid="stMain"] { background:#F7F7F7 !important; }
html, body { font-family:'Inter',sans-serif !important; }

.header-bar {
    background:#8B1A2B;
    padding:1.4rem 2rem 1rem;
    border-radius:10px;
    border-bottom:4px solid #5C0F1A;
    margin-bottom:1.5rem;
}
.header-bar h1 { font-family:'Playfair Display',serif; color:white; font-size:2rem; margin:0; }
.header-bar p  { color:rgba(255,255,255,0.75); margin:0.3rem 0 0; font-size:0.85rem; }
.badge {
    display:inline-block; background:rgba(255,255,255,0.15); color:white;
    font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase;
    padding:0.2rem 0.8rem; border-radius:999px;
    border:1px solid rgba(255,255,255,0.3); margin-bottom:0.5rem;
}
.card {
    background:white; border:1px solid #E0E0E0; border-radius:10px;
    padding:1.2rem 1.5rem; margin-bottom:1rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);
}
.card-title {
    font-weight:600; font-size:0.82rem; letter-spacing:0.08em;
    text-transform:uppercase; color:#8B1A2B; margin-bottom:0.8rem;
    padding-bottom:0.5rem; border-bottom:2px solid #8B1A2B;
}
.metric-row { display:flex; gap:0.8rem; margin-bottom:1rem; }
.metric-box {
    flex:1; background:white; border:1px solid #E0E0E0;
    border-top:3px solid #8B1A2B; border-radius:8px;
    padding:0.7rem; text-align:center;
}
.metric-val { font-size:1.4rem; font-weight:700; color:#8B1A2B; }
.metric-lbl { font-size:0.72rem; color:#888; margin-top:0.1rem; }
div[data-testid="stButton"] > button {
    background:#8B1A2B !important; color:white !important;
    font-weight:600 !important; border:none !important;
    border-radius:8px !important; padding:0.7rem 2rem !important;
    width:100%; font-size:0.95rem !important;
}
div[data-testid="stButton"] > button:hover { opacity:0.85 !important; }
.result-card {
    background:white; border-radius:12px; padding:1.8rem 2rem;
    margin-top:1.2rem; text-align:center;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
}
.result-card.fraud { border:2px solid #C0392B; border-top:5px solid #C0392B; }
.result-card.legit { border:2px solid #1A7A3C; border-top:5px solid #1A7A3C; }
.result-icon  { font-size:2.5rem; display:block; margin-bottom:0.6rem; }
.result-label { font-family:'Playfair Display',serif; font-size:1.8rem; font-weight:700; }
.result-label.fraud { color:#C0392B; }
.result-label.legit { color:#1A7A3C; }
.proba-track  { background:#F0F0F0; border-radius:6px; height:10px; margin:0.8rem auto; max-width:320px; overflow:hidden; }
.proba-fill   { height:10px; border-radius:6px; }
.result-meta  { color:#888; font-size:0.8rem; margin-top:0.6rem; }
.status-ok    { color:#1A7A3C; font-weight:600; font-size:0.9rem; }
.status-err   { color:#C0392B; font-weight:600; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div class="badge">TP MLOps v2.0 · Vrai modèle · 100k transactions</div>
  <h1>🏦 Fraud Detector</h1>
  <p>Détection de transactions frauduleuses — ROC-AUC 95% · Recall Fraude 87%</p>
</div>
""", unsafe_allow_html=True)

# ── État API ──────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">État de l\'API</div>', unsafe_allow_html=True)
try:
    r = requests.get(f"{API_URL}/health", timeout=60)
    if r.status_code == 200:
        data = r.json()
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-box"><div class="metric-val">✅</div><div class="metric-lbl">API en ligne</div></div>
          <div class="metric-box"><div class="metric-val">{data.get('version','—')}</div><div class="metric-lbl">Version</div></div>
          <div class="metric-box"><div class="metric-val">{data.get('roc_auc',0)*100:.1f}%</div><div class="metric-lbl">ROC-AUC</div></div>
          <div class="metric-box"><div class="metric-val">{data.get('accuracy',0)*100:.1f}%</div><div class="metric-lbl">Accuracy</div></div>
        </div>
        <p class="status-ok">● Modèle chargé et opérationnel</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="status-err">● Erreur API : code {r.status_code}</p>', unsafe_allow_html=True)
except requests.exceptions.ConnectionError:
    st.markdown('<p class="status-err">● API inaccessible</p>', unsafe_allow_html=True)
except requests.exceptions.Timeout:
    st.markdown('<p class="status-err">● API en veille — rechargez la page dans 30 secondes</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── Formulaire ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">Analyser une transaction</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    amount     = st.number_input("Montant (€)", min_value=0.01, value=75.0, step=5.0)
    age        = st.number_input("Age du client", min_value=18, max_value=100, value=35)
    hour       = st.slider("Heure de la transaction", 0, 23, 14)
    days_since = st.number_input("Jours depuis derniere transaction", min_value=0, value=7)

with c2:
    gender_lbl  = st.selectbox("Genre", ["Homme (M)", "Femme (F)"])
    gender      = "M" if "M" in gender_lbl else "F"
    category    = st.selectbox("Categorie", [
        "grocery","food","travel","entertainment",
        "health","electronics","clothing","fuel","online","atm"
    ])
    foreign_lbl = st.selectbox("Transaction a l'etranger", ["Non (0)", "Oui (1)"])
    is_foreign  = 1 if "1" in foreign_lbl else 0
    st.markdown("**Payload envoye a l'API :**")
    st.code(
        f'{{"amount":{amount},"age":{age},"transaction_hour":{hour},'
        f'"days_since_last_txn":{days_since},"gender":"{gender}",'
        f'"category":"{category}","is_foreign_country":{is_foreign}}}',
        language="json"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Prédiction ────────────────────────────────────────────────────────────────
if st.button("Analyser la transaction"):
    payload = {
        "amount": amount, "transaction_hour": hour,
        "days_since_last_txn": days_since, "is_foreign_country": is_foreign,
        "age": age, "gender": gender, "category": category,
    }
    with st.spinner("Analyse en cours... (jusqu'a 60s si l'API sort de veille)"):
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
            if r.status_code == 200:
                res   = r.json()
                pred  = res["prediction"]
                label = res["label"]
                proba = res["probability"]
                css   = "fraud" if pred == 1 else "legit"
                icon  = "🚨" if pred == 1 else "✅"
                bar   = "#C0392B" if pred == 1 else "#1A7A3C"
                st.markdown(f"""
                <div class="result-card {css}">
                  <span class="result-icon">{icon}</span>
                  <div class="result-label {css}">{label}</div>
                  <div style="margin:0.4rem auto 0;max-width:320px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#888;margin-bottom:4px;">
                      <span>Legitime</span><span>Fraude</span>
                    </div>
                    <div class="proba-track">
                      <div class="proba-fill" style="width:{proba*100:.0f}%;background:{bar};"></div>
                    </div>
                    <div style="font-size:0.88rem;color:{bar};font-weight:600;margin-top:4px;">
                      Probabilite de fraude : {proba*100:.1f}%
                    </div>
                  </div>
                  <div class="result-meta">Modele {res['model_version']} · ROC-AUC {res['roc_auc']*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            elif r.status_code == 422:
                st.error("Donnees invalides (Erreur 422) — verifiez les champs")
            else:
                st.error(f"Erreur API : code {r.status_code} — {r.text}")
        except requests.exceptions.ConnectionError:
            st.error("API inaccessible. Verifiez l'URL Render.")
        except requests.exceptions.Timeout:
            st.error("Timeout depasse (60s). Rechargez et reessayez — l'API devrait etre reveillée.")
        except Exception as e:
            st.error(f"Erreur inattendue : {e}")

st.markdown("---")
st.markdown(
    '<p style="color:#AAA;font-size:0.75rem;text-align:center;">'
    'TP MLOps 2025-2026 · Wilfried MBEUMI · Eddie ATINDEHOU · Joel ONANA</p>',
    unsafe_allow_html=True
)
