"""
Chargement de l'artefact modèle — pattern singleton
Patch de compatibilité sklearn 1.5 → 1.6
"""
import joblib
from pathlib import Path

_artifact = None

def load_artifact():
    global _artifact
    if _artifact is None:
        path = Path(__file__).parent.parent / "artifacts" / "model.joblib"
        _artifact = joblib.load(path)

        # ── Patch sklearn 1.5 → 1.6 ──────────────────────────────────────────
        # sklearn 1.6 a supprimé le paramètre multi_class de LogisticRegression.
        # Un modèle entraîné avec 1.5 peut stocker cet attribut ; on le réinjecte
        # si absent pour éviter l'AttributeError lors de predict/predict_proba.
        try:
            clf = _artifact["model"].named_steps["clf"]
            if not hasattr(clf, "multi_class"):
                clf.multi_class = "auto"
        except Exception:
            pass

    return _artifact
