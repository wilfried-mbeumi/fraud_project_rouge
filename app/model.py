"""
Chargement de l'artefact modèle — pattern singleton
"""
import joblib
from pathlib import Path

_artifact = None

def load_artifact():
    global _artifact
    if _artifact is None:
        path = Path(__file__).parent.parent / "artifacts" / "model.joblib"
        _artifact = joblib.load(path)
    return _artifact
