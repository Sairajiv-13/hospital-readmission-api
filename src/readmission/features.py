"""Feature engineering shared by training and serving.

The cleaning rules here follow the approach documented for this dataset in
Strack et al. (2014) and are applied identically at train time and at
predict time, so the API can never drift from the training features.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TARGET_COLUMN = "readmitted"

# Dropped because of missingness in the raw file (measured on the UCI
# download): weight 96.9%, max_glu_serum 94.7%, A1Cresult 83.3%,
# medical_specialty 49.1%, payer_code 39.6% missing.
HIGH_MISSING_COLUMNS = ["weight", "max_glu_serum", "A1Cresult", "medical_specialty", "payer_code"]

# Identifiers, not clinical signal.
ID_COLUMNS = ["encounter_id", "patient_nbr"]

# Discharge codes where a 30-day readmission is impossible or not the
# outcome of interest: 11 expired, 13/14 hospice, 19/20/21 expired variants.
EXCLUDED_DISCHARGE_CODES = {11, 13, 14, 19, 20, 21}

MEDICATION_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

MEDICATION_VALUES = {"No", "Steady", "Up", "Down"}

NUMERIC_FEATURES = [
    "age_midpoint",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "service_utilization",
]

BASE_CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    "change",
    "diabetesMed",
]

_AGE_PATTERN = re.compile(r"\[(\d+)-(\d+)\)")


def age_to_midpoint(age: str) -> float:
    """Convert an age bracket such as '[70-80)' to its midpoint, 75."""
    if isinstance(age, str):
        match = _AGE_PATTERN.match(age.strip())
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            return (low + high) / 2.0
        if age.strip().startswith("[100"):
            return 100.0
    return np.nan


def group_diag(code: object) -> str:
    """Group an ICD-9 diagnosis code into a clinical chapter.

    Grouping follows the chapters used in the dataset's source paper.
    Letter-prefixed codes (E and V codes) and anything unmapped fall into
    'Other'; the diabetes code 250.xx gets its own group.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    text = str(code).strip()
    if text in ("", "?", "nan"):
        return "Missing"
    if text.startswith("250"):
        return "Diabetes"
    try:
        value = float(text)
    except ValueError:
        return "Other"
    if 390 <= value <= 434 or 440 <= value <= 459:
        return "Circulatory"
    if 460 <= value <= 519:
        return "Respiratory"
    if 520 <= value <= 579:
        return "Digestive"
    if 580 <= value <= 629:
        return "Genitourinary"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 800 <= value <= 999:
        return "Injury and poisoning"
    if 140 <= value <= 239:
        return "Neoplasm"
    return "Other"


def active_medication_columns(df: pd.DataFrame, max_top_share: float = 0.995) -> list[str]:
    """Medication columns that carry signal.

    Several medication columns are constant in practice (examide and
    citoglipton are 'No' for every encounter). A column whose most common
    value covers at least `max_top_share` of rows is dropped.
    """
    kept = []
    for col in MEDICATION_COLUMNS:
        if col not in df.columns:
            continue
        top_share = df[col].value_counts(normalize=True, dropna=True).max()
        if top_share < max_top_share:
            kept.append(col)
    return kept


def clean_encounters(df: pd.DataFrame, med_columns: list[str] | None = None) -> pd.DataFrame:
    """Apply the documented cleaning rules to raw encounter rows.

    Returns a frame with the model features plus the binary target
    `readmitted_30d` (1 when the raw label is '<30'). Rows whose discharge
    disposition means the patient died or entered hospice are removed.
    """
    work = df.copy()
    work = work.replace("?", np.nan)

    if "discharge_disposition_id" in work.columns:
        codes = pd.to_numeric(work["discharge_disposition_id"], errors="coerce")
        work = work[~codes.isin(EXCLUDED_DISCHARGE_CODES)]

    if med_columns is None:
        med_columns = active_medication_columns(work)

    out = pd.DataFrame(index=work.index)
    out["age_midpoint"] = work["age"].map(age_to_midpoint)
    for col in [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]:
        out[col] = pd.to_numeric(work[col], errors="coerce")
    out["service_utilization"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )

    out["race"] = work["race"].fillna("Unknown").astype(str)
    out["gender"] = work["gender"].fillna("Unknown").replace({"Unknown/Invalid": "Unknown"}).astype(str)
    for col in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        out[col] = pd.to_numeric(work[col], errors="coerce").fillna(-1).astype(int).astype(str)
    out["diag_1_group"] = work["diag_1"].map(group_diag)
    out["diag_2_group"] = work["diag_2"].map(group_diag)
    out["diag_3_group"] = work["diag_3"].map(group_diag)
    out["change"] = work["change"].fillna("No").astype(str)
    out["diabetesMed"] = work["diabetesMed"].fillna("No").astype(str)
    for col in med_columns:
        out[col] = work[col].fillna("No").astype(str)

    if TARGET_COLUMN in work.columns:
        out["readmitted_30d"] = (work[TARGET_COLUMN] == "<30").astype(int)
    return out


def feature_columns(med_columns: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature lists for a medication set."""
    return list(NUMERIC_FEATURES), list(BASE_CATEGORICAL_FEATURES) + list(med_columns)
