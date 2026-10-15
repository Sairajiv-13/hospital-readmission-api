import numpy as np
import pandas as pd
import pytest

from readmission.drift import drift_level, drift_report, population_stability_index


def test_psi_identical_distributions_is_near_zero() -> None:
    rng = np.random.default_rng(42)
    sample = rng.normal(50, 10, size=5000)
    psi = population_stability_index(sample, sample.copy())
    assert psi < 0.01


def test_psi_shifted_distribution_is_significant() -> None:
    rng = np.random.default_rng(42)
    reference = rng.normal(50, 10, size=5000)
    shifted = rng.normal(70, 10, size=5000)
    psi = population_stability_index(reference, shifted)
    assert psi > 0.25
    assert drift_level(psi) == "significant"


def test_psi_small_shift_stays_low() -> None:
    rng = np.random.default_rng(1)
    reference = rng.normal(50, 10, size=5000)
    current = rng.normal(51, 10, size=5000)
    psi = population_stability_index(reference, current)
    assert psi < 0.1
    assert drift_level(psi) == "stable"


def test_psi_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        population_stability_index(np.array([]), np.array([1.0, 2.0]))


def test_drift_report_covers_shared_numeric_columns() -> None:
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(5, 1, 500)})
    current = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(9, 1, 500)})
    report = drift_report(reference, current)
    assert set(report) == {"a", "b"}
    assert report["b"] > report["a"]
