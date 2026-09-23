"""Run the whole analysis end to end.

    python src/train.py

Prints a report to the terminal, writes figures to reports/, writes a summary to
reports/results.md, and saves the final model to models/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import evaluate as ev
from config import (
    LEAKY_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from data import clean, column_types, load_raw, split


def rule(title: str):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def build_pipeline(X, model):
    """Preprocessing + model as one object.

    Everything the model needs done to the data lives inside the pipeline, so the
    exact same steps are applied to train, validation and test. Doing these
    transformations by hand on the whole dataset first is one of the most common
    ways to leak information from the test set into training.
    """
    categorical, numeric = column_types(X)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numeric),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def fit_and_score(X_train, y_train, X_test, y_test, model, threshold=0.5):
    pipe = build_pipeline(X_train, model)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    return pipe, proba, ev.metrics_at(y_test, proba, threshold)


def line(label, m):
    print(f"  {label:<34} recall {m['recall']:6.1%}   precision {m['precision']:6.1%}"
          f"   PR-AUC {m['pr_auc']:.3f}   accuracy {m['accuracy']:6.1%}")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    df = clean(load_raw())

    # ---------------------------------------------------------------- 1
    rule("1. THE DATA AND THE IMBALANCE")
    counts = df[TARGET].value_counts()
    rate = df[TARGET].mean()
    print(f"  Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"  Subscribed (yes): {counts[1]:,}  ({rate:.1%})")
    print(f"  Declined   (no ): {counts[0]:,}  ({1 - rate:.1%})")
    print(f"\n  A model that always predicts 'no' scores {1 - rate:.1%} accuracy")
    print("  while finding zero subscribers. This is why accuracy is not the metric.")

    # ---------------------------------------------------------------- 2
    rule("2. THE DUMB BASELINE")
    Xtr, Xva, Xte, ytr, yva, yte = split(df, drop_leaky=True)
    print(f"  Split: {len(Xtr):,} train / {len(Xva):,} validation / {len(Xte):,} test")

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(Xtr, ytr)
    dummy_proba = dummy.predict_proba(Xte)[:, 1]
    m_dummy = ev.metrics_at(yte, dummy_proba, 0.5)
    line("Always predicts 'no'", m_dummy)
    print(f"  Cost of its mistakes: {m_dummy['cost']:,}")
    print("\n  Any real model must beat this. Accuracy alone cannot tell you whether it has.")

    # ---------------------------------------------------------------- 3
    rule("3. FINDING THE DATA LEAKAGE")
    Xtr_l, Xva_l, Xte_l, ytr_l, yva_l, yte_l = split(df, drop_leaky=False)

    _, proba_leak, m_leak = fit_and_score(
        Xtr_l, ytr_l, Xte_l, yte_l, LogisticRegression(max_iter=1000))
    line("Logistic regression WITH duration", m_leak)

    single = roc_auc_score(yte_l, Xte_l["duration"])
    print(f"\n  Alarm bell: `duration` ALONE, with no model at all, scores ROC-AUC {single:.3f}.")
    print("  One column carrying almost the whole signal is the classic symptom of leakage.")
    print("\n  `duration` is how many seconds the sales call lasted.")
    print("  You only know it AFTER the call has happened — and by then you already")
    print("  know the answer. It cannot exist at the moment you decide who to call.")
    print("  A model using it would score beautifully in a notebook and be useless in a bank.")

    _, proba_clean, m_clean = fit_and_score(
        Xtr, ytr, Xte, yte, LogisticRegression(max_iter=1000))
    line("Logistic regression WITHOUT duration", m_clean)
    ev.plot_leakage_impact(m_leak, m_clean)
    print(f"\n  Removing it costs {m_leak['recall'] - m_clean['recall']:.1%} recall."
          " That drop is the project's most honest number.")

    # ---------------------------------------------------------------- 4
    rule("4. TUNING THE DECISION THRESHOLD")
    pipe_lr = build_pipeline(Xtr, LogisticRegression(max_iter=1000))
    pipe_lr.fit(Xtr, ytr)
    proba_val = pipe_lr.predict_proba(Xva)[:, 1]

    chosen = ev.cheapest_threshold(yva, proba_val)
    print(f"  Chosen on the VALIDATION set (test set untouched): {chosen['threshold']:.2f}")
    print(f"  0.50 is only a default. With a missed subscriber costing 20x a wasted")
    print(f"  call, it is worth calling far more people than 0.50 would suggest.")

    proba_test = pipe_lr.predict_proba(Xte)[:, 1]
    m_default = ev.metrics_at(yte, proba_test, 0.50)
    m_tuned = ev.metrics_at(yte, proba_test, chosen["threshold"])

    print()
    line("Logistic @ 0.50 (default)", m_default)
    line(f"Logistic @ {chosen['threshold']:.2f} (tuned)", m_tuned)
    print(f"\n  Cost at 0.50: {m_default['cost']:,}")
    print(f"  Cost tuned  : {m_tuned['cost']:,}   "
          f"({(1 - m_tuned['cost'] / m_default['cost']):.1%} cheaper)")
    print(f"  Accuracy actually FELL from {m_default['accuracy']:.1%} to {m_tuned['accuracy']:.1%}")
    print("  while the model got more useful. That is the whole argument in one line.")

    sweep_test = ev.sweep_thresholds(yte, proba_test)
    ev.plot_threshold_tradeoff(sweep_test, ev.metrics_at(yte, proba_test, chosen["threshold"]))
    ev.plot_pr_curve(yte, proba_test, m_tuned, baseline_rate=yte.mean())
    ev.plot_confusion(m_default, m_tuned)

    # ---------------------------------------------------------------- 5
    rule("5. A SECOND MODEL, COMPARED FAIRLY")
    pipe_gb = build_pipeline(Xtr, HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    pipe_gb.fit(Xtr, ytr)
    gb_val = pipe_gb.predict_proba(Xva)[:, 1]
    gb_chosen = ev.cheapest_threshold(yva, gb_val)
    gb_test = pipe_gb.predict_proba(Xte)[:, 1]
    m_gb = ev.metrics_at(yte, gb_test, gb_chosen["threshold"])

    print("  Both models get the same treatment: same split, same cost model,")
    print("  threshold tuned on validation, judged once on test.\n")
    line("Baseline (always 'no')", m_dummy)
    line(f"Logistic regression @ {chosen['threshold']:.2f}", m_tuned)
    line(f"Gradient boosting   @ {gb_chosen['threshold']:.2f}", m_gb)
    print(f"\n  Cost — baseline: {m_dummy['cost']:,}   logistic: {m_tuned['cost']:,}"
          f"   boosting: {m_gb['cost']:,}")

    best_name, best_pipe, best_thr, best_m = (
        ("gradient boosting", pipe_gb, gb_chosen["threshold"], m_gb)
        if m_gb["cost"] < m_tuned["cost"]
        else ("logistic regression", pipe_lr, chosen["threshold"], m_tuned)
    )
    print(f"  Winner on cost: {best_name}")

    # ---------------------------------------------------------------- 6
    rule("6. IS THE RESULT REAL, OR A LUCKY SPLIT?")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        build_pipeline(Xtr, LogisticRegression(max_iter=1000)),
        Xtr, ytr, cv=cv, scoring="average_precision")
    print(f"  5-fold PR-AUC on the training data: {scores.mean():.3f} "
          f"(+/- {scores.std():.3f})")
    print(f"  Test-set PR-AUC: {m_tuned['pr_auc']:.3f}")
    print("  Close together and a small spread means the number is not a fluke.")

    # ---------------------------------------------------------------- 7
    rule("7. SAVED OUTPUT")
    joblib.dump({"pipeline": best_pipe, "threshold": best_thr}, MODELS_DIR / "final_model.joblib")
    write_results(df, m_dummy, m_leak, m_clean, m_default, m_tuned, m_gb,
                  chosen["threshold"], gb_chosen["threshold"], single, scores, best_name)
    for f in sorted(REPORTS_DIR.glob("*")):
        print(f"  reports/{f.name}")
    print("  models/final_model.joblib")


def write_results(df, m_dummy, m_leak, m_clean, m_default, m_tuned, m_gb,
                  thr_lr, thr_gb, single_auc, cv_scores, best_name):
    rate = df[TARGET].mean()
    rows = [
        ("Baseline — always predicts 'no'", m_dummy, "0.50"),
        ("Logistic regression, default threshold", m_default, "0.50"),
        ("Logistic regression, tuned threshold", m_tuned, f"{thr_lr:.2f}"),
        ("Gradient boosting, tuned threshold", m_gb, f"{thr_gb:.2f}"),
    ]
    table = "\n".join(
        f"| {name} | {t} | {m['accuracy']:.1%} | {m['precision']:.1%} | "
        f"{m['recall']:.1%} | {m['pr_auc']:.3f} | {m['cost']:,} |"
        for name, m, t in rows
    )

    (REPORTS_DIR / "results.md").write_text(f"""# Results

Generated by `python src/train.py`. All figures are on the held-out test set,
which was split off before any model was fitted and used exactly once.

## Class balance

{len(df):,} customers, of whom {df[TARGET].sum():,} subscribed ({rate:.1%}).
Predicting "no" for everyone therefore scores **{1 - rate:.1%} accuracy** while
finding nobody.

## Model comparison

| Model | Threshold | Accuracy | Precision | Recall | PR-AUC | Cost of mistakes |
|---|---|---|---|---|---|---|
{table}

Winner on business cost: **{best_name}**.

## Data leakage

`duration` (length of the sales call in seconds) is not known before the call is
made, so it cannot be an input to a decision about whom to call. On its own it
scores ROC-AUC **{single_auc:.3f}** against the target.

| | Recall | Precision | PR-AUC |
|---|---|---|---|
| With `duration` | {m_leak['recall']:.1%} | {m_leak['precision']:.1%} | {m_leak['pr_auc']:.3f} |
| Without `duration` | {m_clean['recall']:.1%} | {m_clean['precision']:.1%} | {m_clean['pr_auc']:.3f} |

## Stability

5-fold cross-validated PR-AUC on the training data:
**{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}**, against a test-set PR-AUC of
{m_tuned['pr_auc']:.3f}.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
