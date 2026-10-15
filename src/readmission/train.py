"""Train the readmission model and write versioned artifacts.

Usage:
    python -m readmission.train --data data/diabetic_data.csv --artifacts artifacts

The split is grouped by patient: encounters from the same patient never
appear on both sides, so the reported metrics are not inflated by
memorized patients.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from readmission import __version__
from readmission.features import (
    MEDICATION_COLUMNS,
    clean_encounters,
    feature_columns,
)

MODEL_VERSION = __version__
RANDOM_STATE = 42
# Recall the tuned decision threshold aims for on validation data. Half of
# the patients who come back inside 30 days is a reasonable catch rate for
# a screening flag; the precision cost is reported, not hidden.
TARGET_RECALL = 0.5


def build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def build_candidates(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    logistic = Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical, scale=True)),
            (
                "model",
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            ),
        ]
    )
    boosting = Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical, scale=False)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    candidates = {"logistic_regression": logistic, "hist_gradient_boosting": boosting}
    try:
        from lightgbm import LGBMClassifier

        candidates["lightgbm"] = Pipeline(
            [
                ("preprocess", build_preprocessor(numeric, categorical, scale=False)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=31,
                        min_child_samples=40,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        pass
    return candidates


def classification_scores(probabilities: np.ndarray, y: pd.Series, threshold: float) -> dict:
    """Confusion-matrix scores for one fixed probability threshold."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "precision": round(float(precision_score(y, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "positive_rate": round(float(y.mean()), 4),
    }


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    probabilities = model.predict_proba(x)[:, 1]
    return {
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4),
        **classification_scores(probabilities, y, 0.5),
    }


def tune_threshold(probabilities: np.ndarray, y: pd.Series, target_recall: float) -> float:
    """Pick the grid threshold whose recall lands closest to the target.

    The grid runs from 0.05 to 0.95 in steps of 0.01. Ties go to the
    threshold with the higher precision, then to the lower threshold.
    """
    best_threshold = 0.5
    best_key: tuple[float, float, float] | None = None
    for step in range(5, 96):
        threshold = step / 100
        scores = classification_scores(probabilities, y, threshold)
        key = (abs(scores["recall"] - target_recall), -scores["precision"], threshold)
        if best_key is None or key < best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def threshold_analysis(
    winner: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Threshold behaviour of the winning model, measured on held-out data.

    The 0.5 default is the wrong operating point for this label: only
    about 11 percent of encounters are positive, and the boosting models
    put almost no probability mass above 0.5, so recall collapses. The
    tuned threshold is picked on a patient-grouped validation slice of
    the training rows. A clone of the winner is fitted without those
    rows, so the tuning model never sees its own validation data. The
    shipped model (fitted on all training rows) is then scored once on
    the held-out test patients at 0.3, at 0.5, and at the tuned value.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    fit_idx, val_idx = next(splitter.split(x_train, y_train, train_groups))
    tuning_model = clone(winner)
    tuning_model.fit(x_train.iloc[fit_idx], y_train.iloc[fit_idx])
    y_val = y_train.iloc[val_idx]
    val_probabilities = tuning_model.predict_proba(x_train.iloc[val_idx])[:, 1]
    tuned = tune_threshold(val_probabilities, y_val, TARGET_RECALL)

    test_probabilities = winner.predict_proba(x_test)[:, 1]
    results = {}
    for label, threshold in (("0.3", 0.3), ("0.5", 0.5), ("tuned", tuned)):
        results[label] = {
            "threshold": round(float(threshold), 4),
            **classification_scores(test_probabilities, y_test, threshold),
        }
    return {
        "target_recall": TARGET_RECALL,
        "tuned_threshold": round(float(tuned), 4),
        "validation_rows": int(len(val_idx)),
        "validation_recall_at_tuned_threshold": classification_scores(
            val_probabilities, y_val, tuned
        )["recall"],
        "test_results": results,
        "method": (
            "Tuned threshold picked on a patient-grouped validation slice "
            "(20 percent) of the training rows, targeting recall 0.5, using "
            "a clone of the winner fitted without the validation rows. All "
            "reported threshold metrics are measured on the held-out test "
            "patients."
        ),
    }


def top_features(model: Pipeline, x: pd.DataFrame, y: pd.Series, limit: int = 10) -> list[dict]:
    """Global feature ranking for the winning model.

    Logistic regression: absolute standardized coefficients. Tree model:
    permutation importance on the test set (capped for runtime). Scores
    from the two methods are not comparable to each other; the card
    records which method produced them.
    """
    estimator = model.named_steps["model"]
    names = list(model.named_steps["preprocess"].get_feature_names_out())
    if isinstance(estimator, LogisticRegression):
        scores = np.abs(estimator.coef_[0])
        method = "absolute standardized coefficient"
    else:
        sample = min(len(x), 5000)
        result = permutation_importance(
            model, x.iloc[:sample], y.iloc[:sample], n_repeats=3, random_state=RANDOM_STATE, scoring="roc_auc"
        )
        scores = result.importances_mean
        method = "permutation importance (roc_auc drop)"
    order = np.argsort(scores)[::-1][:limit]
    return [
        {"feature": names[i], "score": round(float(scores[i]), 4), "method": method} for i in order
    ]


def train(data_path: Path, artifacts_dir: Path) -> dict:
    raw = pd.read_csv(data_path, low_memory=False)
    rows_raw = int(len(raw))
    patients = raw["patient_nbr"] if "patient_nbr" in raw.columns else pd.Series(index=raw.index)

    cleaned = clean_encounters(raw)
    rows_after_cleaning = int(len(cleaned))
    med_columns = [col for col in cleaned.columns if col in MEDICATION_COLUMNS]
    numeric, categorical = feature_columns(med_columns)
    features = numeric + categorical

    x = cleaned[features]
    y = cleaned["readmitted_30d"]
    groups = patients.loc[cleaned.index]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(x, y, groups))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    candidates = build_candidates(numeric, categorical)
    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        results[name] = evaluate(pipeline, x_test, y_test)
        fitted[name] = pipeline

    winner_name = max(results, key=lambda name: results[name]["roc_auc"])
    winner = fitted[winner_name]
    analysis = threshold_analysis(winner, x_train, y_train, groups.iloc[train_idx], x_test, y_test)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner, artifacts_dir / "model.joblib")

    reference = x_train[numeric].sample(n=min(2000, len(x_train)), random_state=RANDOM_STATE)
    reference.to_json(artifacts_dir / "reference_sample.json", orient="records")

    card = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset": "UCI Diabetes 130-US Hospitals for Years 1999-2008",
        "rows_raw": rows_raw,
        "rows_after_cleaning": rows_after_cleaning,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "split": "GroupShuffleSplit by patient_nbr, test_size 0.2, random_state 42",
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "medication_features": med_columns,
        "models": results,
        "selected_model": winner_name,
        "selected_metrics": results[winner_name],
        "decision_threshold": analysis["tuned_threshold"],
        "threshold_analysis": analysis,
        "top_features": top_features(winner, x_test, y_test),
        "intended_use": "Portfolio demonstration of an ML serving pipeline. Not for clinical use.",
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(card, indent=2))
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the readmission model.")
    parser.add_argument("--data", type=Path, default=Path("data/diabetic_data.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    card = train(args.data, args.artifacts)
    print(json.dumps({"selected_model": card["selected_model"], **card["selected_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
