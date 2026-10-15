"""FastAPI serving layer for the readmission model.

Run locally with:
    uvicorn readmission.api:app --host 0.0.0.0 --port 8000

Artifacts are read from the directory named by the ARTIFACTS_DIR
environment variable (default: ./artifacts). The model is loaded once at
startup; if artifacts are missing, /healthz still answers and /predict
returns 503 until a model is trained.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.linear_model import LogisticRegression

from readmission import __version__
from readmission.features import MEDICATION_VALUES, age_to_midpoint, clean_encounters

MAX_BATCH_SIZE = 100


def risk_band(probability: float) -> str:
    """Heuristic bands around the dataset base rate of about 0.11.

    These are display bands for triage demos, not clinical thresholds.
    """
    if probability < 0.10:
        return "low"
    if probability < 0.20:
        return "moderate"
    return "high"


class EncounterInput(BaseModel):
    """One hospital encounter, in the raw dataset's terms."""

    age: str = Field(examples=["[70-80)"])
    race: str = "Unknown"
    gender: str = "Unknown"
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int = Field(ge=1, le=14)
    num_lab_procedures: int = Field(default=0, ge=0)
    num_procedures: int = Field(default=0, ge=0)
    num_medications: int = Field(ge=0)
    number_outpatient: int = Field(default=0, ge=0)
    number_emergency: int = Field(default=0, ge=0)
    number_inpatient: int = Field(default=0, ge=0)
    number_diagnoses: int = Field(default=1, ge=1, le=16)
    diag_1: str
    diag_2: str | None = None
    diag_3: str | None = None
    change: str = "No"
    diabetes_med: str = Field(default="No", alias="diabetesMed")
    medications: dict[str, str] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @field_validator("age")
    @classmethod
    def age_must_be_a_bracket(cls, value: str) -> str:
        if np.isnan(age_to_midpoint(value)):
            raise ValueError("age must be a bracket such as '[70-80)'")
        return value

    @field_validator("medications")
    @classmethod
    def medication_values_must_be_known(cls, value: dict[str, str]) -> dict[str, str]:
        bad = {k: v for k, v in value.items() if v not in MEDICATION_VALUES}
        if bad:
            raise ValueError(f"medication values must be one of {sorted(MEDICATION_VALUES)}: {bad}")
        return value

    @model_validator(mode="after")
    def medication_names_checked_against_model(self) -> EncounterInput:
        known = set(_store.card.get("medication_features", [])) if _store.card else None
        if known is not None:
            unknown = set(self.medications) - known
            if unknown:
                raise ValueError(f"unknown medication fields for this model: {sorted(unknown)}")
        return self

    def to_raw_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "age": self.age,
            "race": self.race,
            "gender": self.gender,
            "admission_type_id": self.admission_type_id,
            "discharge_disposition_id": self.discharge_disposition_id,
            "admission_source_id": self.admission_source_id,
            "time_in_hospital": self.time_in_hospital,
            "num_lab_procedures": self.num_lab_procedures,
            "num_procedures": self.num_procedures,
            "num_medications": self.num_medications,
            "number_outpatient": self.number_outpatient,
            "number_emergency": self.number_emergency,
            "number_inpatient": self.number_inpatient,
            "number_diagnoses": self.number_diagnoses,
            "diag_1": self.diag_1,
            "diag_2": self.diag_2 if self.diag_2 is not None else "?",
            "diag_3": self.diag_3 if self.diag_3 is not None else "?",
            "change": self.change,
            "diabetesMed": self.diabetes_med,
        }
        for med in _store.card.get("medication_features", []):
            row[med] = self.medications.get(med, "No")
        return row


class BatchInput(BaseModel):
    encounters: list[EncounterInput] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    readmission_probability: float
    high_risk: bool
    risk_band: str
    model_version: str
    attribution_method: str
    top_contributing_features: list[dict[str, Any]]


class ModelStore:
    """Loads and holds the trained pipeline plus its model card."""

    def __init__(self) -> None:
        self.model: Any = None
        self.card: dict[str, Any] = {}

    def load(self) -> None:
        artifacts = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
        model_path = artifacts / "model.joblib"
        card_path = artifacts / "metrics.json"
        if model_path.exists() and card_path.exists():
            self.model = joblib.load(model_path)
            self.card = json.loads(card_path.read_text())

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def decision_threshold(self) -> float:
        """Operating threshold for the high_risk flag, from the model card.

        Set at training time by tuning on a validation split for recall
        near 0.5 (see threshold_analysis in metrics.json). Falls back to
        0.5 for cards written before threshold tuning existed.
        """
        value = self.card.get("decision_threshold")
        return float(value) if value is not None else 0.5


_store = ModelStore()


def _feature_frame(encounters: list[EncounterInput]) -> pd.DataFrame:
    raw = pd.DataFrame([item.to_raw_row() for item in encounters])
    med_columns = list(_store.card.get("medication_features", []))
    cleaned = clean_encounters(raw, med_columns=med_columns)
    if len(cleaned) != len(encounters):
        raise HTTPException(
            status_code=422,
            detail=(
                "One or more encounters use a discharge disposition that is "
                "out of scope for this model (death or hospice discharge)."
            ),
        )
    return cleaned[_store.card["features"]]


def _local_attributions(frame: pd.DataFrame) -> tuple[str, list[dict[str, Any]]]:
    """Per-request attributions where the model supports them.

    Logistic regression exposes exact per-feature contributions. The
    gradient boosting model does not, so the API returns the global
    permutation importances stored in the model card and labels them as
    such instead of inventing local values.
    """
    model = _store.model
    estimator = model.named_steps["model"]
    if isinstance(estimator, LogisticRegression):
        preprocessor = model.named_steps["preprocess"]
        names = [name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()]
        vector = np.asarray(preprocessor.transform(frame))[0]
        contributions = vector * estimator.coef_[0]
        order = np.argsort(np.abs(contributions))[::-1]
        top = [
            {"feature": names[i], "contribution": round(float(contributions[i]), 4)}
            for i in order
            if contributions[i] != 0
        ][:5]
        return "local_coefficient_contribution", top
    return "global_permutation_importance", list(_store.card.get("top_features", []))[:5]


def _predict(encounters: list[EncounterInput]) -> list[Prediction]:
    if not _store.ready:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    frame = _feature_frame(encounters)
    probabilities = _store.model.predict_proba(frame)[:, 1]
    method, top = _local_attributions(frame.iloc[[0]])
    version = _store.card.get("model_version", __version__)
    threshold = _store.decision_threshold
    return [
        Prediction(
            readmission_probability=round(float(p), 4),
            high_risk=bool(p >= threshold),
            risk_band=risk_band(float(p)),
            model_version=version,
            attribution_method=method,
            top_contributing_features=top if len(encounters) == 1 else [],
        )
        for p in probabilities
    ]


@asynccontextmanager
async def lifespan(_: FastAPI):
    _store.load()
    yield


app = FastAPI(title="Hospital Readmission Risk API", version=__version__, lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _store.ready,
        "model_version": _store.card.get("model_version") if _store.ready else None,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if not _store.ready:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    card = _store.card
    return {
        "model_version": card.get("model_version"),
        "selected_model": card.get("selected_model"),
        "trained_at": card.get("trained_at"),
        "dataset": card.get("dataset"),
        "train_rows": card.get("train_rows"),
        "test_rows": card.get("test_rows"),
        "metrics": card.get("selected_metrics"),
        "decision_threshold": _store.decision_threshold,
        "threshold_analysis": card.get("threshold_analysis"),
        "all_models": card.get("models"),
        "features": card.get("features"),
        "intended_use": card.get("intended_use"),
    }


@app.post("/predict", response_model=Prediction)
def predict(encounter: EncounterInput) -> Prediction:
    return _predict([encounter])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: BatchInput) -> list[Prediction]:
    return _predict(batch.encounters)
