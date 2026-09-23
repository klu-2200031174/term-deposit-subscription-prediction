"""Score customers with the saved model.

    python src/predict.py

Loads models/final_model.joblib (created by train.py), scores the first 10 rows
of the dataset, and prints who the model would call.

This exists to prove the saved model is usable on its own, without rerunning the
whole analysis — which is what "deploying" a model means at the smallest scale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib

from config import LEAKY_COLUMNS, MODELS_DIR, TARGET
from data import clean, load_raw

MODEL_PATH = MODELS_DIR / "final_model.joblib"


def main(n: int = 10):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run `python src/train.py` first.")

    saved = joblib.load(MODEL_PATH)
    pipeline, threshold = saved["pipeline"], saved["threshold"]

    df = clean(load_raw()).head(n)
    truth = df[TARGET]
    features = df.drop(columns=[TARGET] + LEAKY_COLUMNS)

    proba = pipeline.predict_proba(features)[:, 1]

    print(f"Decision threshold: {threshold:.2f}")
    print(f"{'row':>4}  {'P(subscribe)':>13}  {'decision':>10}  {'actual':>8}")
    print("-" * 42)
    for i, (p, actual) in enumerate(zip(proba, truth)):
        decision = "CALL" if p >= threshold else "skip"
        print(f"{i:>4}  {p:>13.3f}  {decision:>10}  {'yes' if actual else 'no':>8}")


if __name__ == "__main__":
    main()
