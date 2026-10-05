"""
model.py

Trains the "improved method" classifier (supervised ML + SHAP
explainability, per the capstone's baseline-vs-improved-method
structure) on the engineered FT features, evaluates it on the held-out
lot-level test set, and explains its predictions.

Model choice
------------
RandomForestClassifier:
  - Works well on small-to-moderate tabular data (1,094 train rows here)
    without heavy tuning.
  - Handles the mixed scale of features (mm, %, mA, V, ms, 0/1 flags)
    without needing feature scaling.
  - class_weight="balanced" compensates for the class imbalance left
    after category consolidation (726 Dimensional/Alignment vs 28
    Other/Minor in the raw counts; weighting re-balances the loss rather
    than discarding majority-class data).
  - Pairs directly with SHAP's TreeExplainer for exact (not approximate)
    feature attributions, which is what lets us say *why* the model
    predicted a given defect family from the FT readings -- the actual
    root-cause-identification goal of the capstone, not just a
    scrap/no-scrap flag.

Hyperparameter search
----------------------
A small grid search over a few RandomForest settings, cross-validated
with GroupKFold on the TRAINING set only (grouped by PO), so no lot ever
appears in both the fit and validation fold of a single CV split --
consistent with how train_test_split.py avoided lot leakage between
train and test. The held-out test set is touched exactly once, for the
final evaluation below.

Evaluation
----------
Plain accuracy is a poor summary here given the class imbalance (a model
that always predicts "Dimensional / Alignment" would score ~53% without
learning anything), so this reports:
  - Per-class precision/recall/F1 (classification_report)
  - Macro F1 (unweighted average across classes -- penalises ignoring
    small classes, unlike micro/weighted averages)
  - A confusion matrix plot

Explainability
--------------
SHAP TreeExplainer computed on the test set, saved as:
  - A global summary plot (which FT parameters matter most, overall)
  - Per-class bar plots (which parameters push toward each specific
    defect family) -- this is the artifact that answers "how did the
    model decide this was a Coil defect vs a Dimensional/Alignment one?"

Usage
-----
    python src/model.py \\
        --train data/processed/train_features.csv \\
        --test data/processed/test_features.csv \\
        --out-dir reports/figures \\
        --model-out models/defect_classifier.joblib
"""

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")  # headless -- just save PNGs, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV


def load_features(path: str):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in ("PO", "model_target")]
    X = df[feature_cols]
    y = df["model_target"]
    groups = df["PO"]
    return X, y, groups, feature_cols


def train_model(X_train, y_train, groups_train, seed: int):
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3],
    }
    base = RandomForestClassifier(class_weight="balanced", random_state=seed)
    cv = GroupKFold(n_splits=5)
    # Materialize the split into a concrete list of (train_idx, test_idx)
    # tuples rather than passing the generator cv.split(...) directly.
    # GridSearchCV with n_jobs=-1 pickles the CV splits to ship to worker
    # processes (via joblib/loky), and generators aren't picklable -- this
    # fails with "cannot pickle 'generator' object" on Windows (and on
    # Linux too, once the dataset is large enough that multiprocessing
    # actually kicks in instead of running inline).
    cv_splits = list(cv.split(X_train, y_train, groups=groups_train))
    search = GridSearchCV(
        base,
        param_grid,
        scoring="f1_macro",
        cv=cv_splits,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best params (by group-CV macro F1 on train): {search.best_params_}")
    print(f"Best CV macro F1: {search.best_score_:.3f}\n")
    return search.best_estimator_


def evaluate(model, X_test, y_test, out_dir: str):
    y_pred = model.predict(X_test)

    print("=== Classification report (test set) ===")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1 (unweighted across classes): {macro_f1:.3f}")
    print("(Plain accuracy is misleading here given class imbalance -- "
          "macro F1 is the headline number to report/defend.)\n")

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nMacro F1: {macro_f1:.3f}\n")

    labels = sorted(y_test.unique())
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=labels, xticks_rotation=45, ax=ax, colorbar=False
    )
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix -> {cm_path}")

    return y_pred, macro_f1


def explain(model, X_test, feature_cols, out_dir: str):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    classes = list(model.classes_)

    # Global summary: mean |SHAP value| per feature, averaged across all
    # classes -- which FT parameters matter most overall.
    fig = plt.figure(figsize=(9, 8))
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_cols, class_names=classes,
        show=False, plot_type="bar",
    )
    plt.tight_layout()
    summary_path = os.path.join(out_dir, "shap_summary_overall.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SHAP overall summary -> {summary_path}")

    # Per-class: which parameters push the model toward THIS specific
    # defect family -- the actual root-cause-explainability deliverable.
    sv = shap_values
    if isinstance(sv, list):
        per_class = {cls: sv[i] for i, cls in enumerate(classes)}
    else:
        # sklearn>=1.4/shap>=0.45 returns a single (n, features, n_classes) array
        per_class = {cls: sv[:, :, i] for i, cls in enumerate(classes)}

    for cls, values in per_class.items():
        fig = plt.figure(figsize=(8, 7))
        shap.summary_plot(
            values, X_test, feature_names=feature_cols, show=False, plot_type="bar"
        )
        plt.title(f"SHAP feature impact -> {cls}")
        plt.tight_layout()
        safe_name = cls.replace(" / ", "_").replace(" ", "_")
        path = os.path.join(out_dir, f"shap_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"SHAP summary for '{cls}' -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--model-out", default="models/defect_classifier.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    X_train, y_train, groups_train, feature_cols = load_features(args.train)
    X_test, y_test, groups_test, _ = load_features(args.test)

    print(f"Train: {len(X_train)} rows across {groups_train.nunique()} POs")
    print(f"Test:  {len(X_test)} rows across {groups_test.nunique()} POs\n")

    model = train_model(X_train, y_train, groups_train, args.seed)

    y_pred, macro_f1 = evaluate(model, X_test, y_test, args.out_dir)

    explain(model, X_test, feature_cols, args.out_dir)

    joblib.dump(model, args.model_out)
    print(f"\nModel saved -> {args.model_out} (gitignored -- rerun this script "
          f"to regenerate rather than committing the binary)")


if __name__ == "__main__":
    main()
