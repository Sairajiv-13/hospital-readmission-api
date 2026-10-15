"""Input drift checks based on the population stability index (PSI).

PSI compares the distribution of a feature in a reference sample (usually
training data) against a current batch. Values under 0.1 are read as no
material shift, 0.1 to 0.25 as a moderate shift worth watching, and above
0.25 as a significant shift. Those thresholds are industry convention, not
a property of this dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPSILON = 1e-6


def population_stability_index(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """PSI of `current` against `reference`, using reference quantile bins."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI needs non-empty reference and current samples")

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        # Near-constant reference: fall back to a two-bin comparison on the
        # single observed value versus everything else.
        edges = np.array([-np.inf, edges[0], np.inf]) if edges.size else np.array([-np.inf, np.inf])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.maximum(ref_counts / ref_counts.sum(), _EPSILON)
    cur_pct = np.maximum(cur_counts / cur_counts.sum(), _EPSILON)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 10,
) -> dict[str, float]:
    """PSI per numeric column present in both frames."""
    if columns is None:
        columns = [
            col
            for col in reference.columns
            if col in current.columns and pd.api.types.is_numeric_dtype(reference[col])
        ]
    return {
        col: population_stability_index(reference[col], current[col], bins=bins) for col in columns
    }


def drift_level(psi: float) -> str:
    """Bucket a PSI value using the conventional thresholds."""
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"
