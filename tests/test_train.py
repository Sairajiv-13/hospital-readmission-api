import json
from pathlib import Path

import joblib

from readmission.train import train

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_encounters.csv"


def test_training_writes_artifacts_and_sane_metrics(tmp_path: Path) -> None:
    card = train(FIXTURE_CSV, tmp_path)

    assert (tmp_path / "model.joblib").exists()
    metrics_file = tmp_path / "metrics.json"
    assert metrics_file.exists()
    on_disk = json.loads(metrics_file.read_text())
    assert on_disk["selected_model"] == card["selected_model"]
    assert on_disk["model_version"] == "1.0.0"

    for model_metrics in card["models"].values():
        assert 0.4 <= model_metrics["roc_auc"] <= 1.0
    assert card["train_rows"] > 0 and card["test_rows"] > 0
    assert card["rows_after_cleaning"] <= card["rows_raw"]
    assert "age_midpoint" in card["features"]
    assert (tmp_path / "reference_sample.json").exists()

    model = joblib.load(tmp_path / "model.joblib")
    assert hasattr(model, "predict_proba")


def test_threshold_analysis_is_recorded(tmp_path: Path) -> None:
    card = train(FIXTURE_CSV, tmp_path)

    analysis = card["threshold_analysis"]
    assert card["decision_threshold"] == analysis["tuned_threshold"]
    assert 0.0 < analysis["tuned_threshold"] < 1.0
    assert analysis["validation_rows"] > 0

    results = analysis["test_results"]
    assert set(results) == {"0.3", "0.5", "tuned"}
    assert results["0.5"]["threshold"] == 0.5
    # The per-model metrics stay at the 0.5 default; the threshold table
    # must agree with them there.
    assert results["0.5"]["recall"] == card["selected_metrics"]["recall"]
    assert results["0.5"]["precision"] == card["selected_metrics"]["precision"]
    # Recall cannot increase as the threshold rises.
    by_threshold = sorted(results.values(), key=lambda row: row["threshold"])
    recalls = [row["recall"] for row in by_threshold]
    assert recalls == sorted(recalls, reverse=True)
    # The tuned value was picked to land near the target recall on the
    # validation rows the tuning model never trained on.
    assert abs(analysis["validation_recall_at_tuned_threshold"] - analysis["target_recall"]) <= 0.25
