# Hospital Readmission Risk API

A small ML service that scores a hospital encounter for 30-day readmission risk and serves the score over HTTP. It is trained on the UCI Diabetes 130-US Hospitals dataset (101,766 encounters from 1999 to 2008), compares three model families on a patient-grouped split, ships the winner behind a FastAPI app, and includes a drift check on the inputs. The trained artifacts are committed, so the API serves predictions as soon as it starts. It is a portfolio project that shows the full path from raw data to a served model. It is not for clinical use.

Live demo: https://hospital-readmission-api-ec05.onrender.com (free tier, the first request after idle can take a minute)

## Problem

About 11 percent of the encounters in this dataset end with the patient back in hospital inside 30 days. Those returns are expensive and often preventable, so a discharge team wants a short list of patients worth a second look: a follow-up call, a medication review, a sooner clinic visit. The catch is the imbalance. A model can look accurate by almost never raising its hand, and the first version of this one did exactly that. The interesting work here is not squeezing out AUC. It is measuring honestly, then picking a decision threshold that makes the flag worth having.

## Approach

Cleaning follows the rules documented for this dataset by Strack et al. (2014). Columns that are mostly missing are dropped (weight is 96.9 percent empty, max_glu_serum 94.7 percent, A1Cresult 83.3 percent), diagnosis codes are grouped into ICD-9 chapters, age brackets become midpoints, and encounters that end in death or hospice discharge are removed because a readmission is not the outcome there. The same cleaning function runs at training time and at predict time, so serving cannot drift away from training features.

The split is a GroupShuffleSplit by patient: 79,541 training rows and 19,802 test rows, with no patient on both sides. Three models train on identical features and are compared on ROC AUC: logistic regression, histogram gradient boosting, and LightGBM. The winner ships. Because the default 0.5 threshold is the wrong operating point for an 11 percent positive rate, the decision threshold is tuned separately on a patient-grouped validation slice of the training rows, targeting recall near 0.5, and stored in the model card. ADR 0001 covers the model choice and ADR 0002 covers the threshold.

Serving is a FastAPI app that loads the pipeline once at startup. Input drift is checked with a population stability index module against a reference sample of the training data.

## Results

All numbers below are on the 19,802 held-out test encounters (positive rate 11.19 percent) and are recorded in artifacts/metrics.json.

Model comparison, with precision, recall, and F1 at the default 0.5 threshold:

| Model | ROC AUC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Logistic regression | 0.6670 | 0.1824 | 0.5420 | 0.2730 |
| Histogram gradient boosting | 0.6685 | 0.5513 | 0.0194 | 0.0375 |
| LightGBM | 0.6700 | 0.5842 | 0.0266 | 0.0509 |

LightGBM won and shipped, by 0.003 AUC over logistic regression. That is a tie for practical purposes, and the README is not going to dress it up. Published work on this dataset lands in the same AUC range; the features recorded at discharge only carry so much signal.

The threshold table is the part that matters. Same model, same test patients, three cutoffs:

| Threshold | Precision | Recall | F1 | What happens |
| --- | --- | --- | --- | --- |
| 0.50 (default) | 0.5842 | 0.0266 | 0.0509 | Catches 59 of 2,216 readmissions |
| 0.30 | 0.4036 | 0.1020 | 0.1628 | Catches 226 of 2,216 |
| 0.13 (tuned) | 0.1925 | 0.4991 | 0.2778 | Catches 1,106 of 2,216, flags 29.0 percent of encounters |

At the default threshold the boosting models look precise and are nearly useless: they put almost no probability mass above 0.5, so recall collapses. The tuned threshold of 0.13 was picked on 15,734 validation rows (recall 0.4972 there) and holds up on the test patients at 0.4991. The price is precision 0.1925, roughly four false alarms for every true catch. For a screening flag that trades staff time against missed readmissions, that is the honest bargain, and it is written down in ADR 0002.

The strongest signals, by permutation importance on the test set (drop in ROC AUC): prior inpatient visits (0.0491), then two race categories (0.0440 and 0.0098), length of stay (0.0093), and total prior service use (0.0073). Prior utilization dominating is expected. Race categories ranking that high says more about the dataset and how it was recorded than about medicine, and it is one more reason this model stays a demo.

## Tech stack

- Python 3.11 or newer
- FastAPI and pydantic for the API
- scikit-learn for preprocessing, the model comparison, and metrics
- LightGBM for the shipped model
- pandas and joblib for data handling and the model artifact
- pytest and ruff for tests and linting
- Docker for packaging, with a render.yaml for a free Render deploy
- Drift checks are a small in-repo PSI implementation, no extra dependency

## Run it locally

The trained artifacts are committed under artifacts/, so the API works without retraining:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/uvicorn readmission.api:app --port 8000
```

Or with make: `make setup`, then `make serve`. Tests and lint: `.venv/bin/pytest` and `.venv/bin/ruff check src tests` (20 tests, 93 percent coverage at the last run). Docker: `docker build -t hospital-readmission-api .` then `docker run -p 8000:8000 hospital-readmission-api`.

To retrain, download the dataset as described in data/README.md, then run `make train`. Retraining reruns the full model comparison and the threshold tuning, and rewrites artifacts/metrics.json, artifacts/model.joblib (about 1.0 MB), and artifacts/reference_sample.json.

## API endpoints

| Endpoint | What it does |
| --- | --- |
| GET /healthz | Service status and whether the model is loaded |
| GET /model-info | Model card: version, dataset, per-model metrics, threshold analysis |
| POST /predict | Score one encounter |
| POST /predict/batch | Score 1 to 100 encounters in one call |

A prediction returns the raw model probability, a high_risk flag, a risk_band display heuristic (low, moderate, high, cut around the base rate), and feature attributions. The flag is the probability at or above the tuned decision_threshold from the model card (0.13 in the current artifacts). The band is display only. Attributions are honest about their kind: the response names its attribution_method, and for the shipped tree model these are the global permutation importances from the model card, not per-patient explanations.

Example, from a local run against the committed artifacts:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age": "[70-80)", "race": "Caucasian", "gender": "Female",
       "admission_type_id": 1, "discharge_disposition_id": 1,
       "admission_source_id": 7, "time_in_hospital": 5,
       "num_lab_procedures": 41, "num_procedures": 1, "num_medications": 12,
       "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 0,
       "number_diagnoses": 8, "diag_1": "414.01", "diag_2": "250.0",
       "diag_3": "401.9", "change": "Ch", "diabetesMed": "Yes",
       "medications": {"insulin": "Steady", "metformin": "Steady"}}'
```

Response (attribution list trimmed here; the API returns five entries):

```json
{
  "readmission_probability": 0.1044,
  "high_risk": false,
  "risk_band": "moderate",
  "model_version": "1.0.0",
  "attribution_method": "global_permutation_importance"
}
```

Encounters whose discharge disposition is death or hospice get a 422: those rows were excluded from training, and the API says so instead of scoring them.

## Model notes and limitations

- The data covers US hospital encounters from 1999 to 2008 for patients with a diabetes diagnosis. Admission patterns, coding habits, and treatments have moved since then. A model trained on it describes that era.
- The label is readmission inside 30 days only. Readmissions on day 31 or later, and deaths after discharge, are counted as negatives.
- ROC AUC 0.67 is modest. The model ranks somewhat better than chance and nowhere near well enough to act on by itself. Any real use would need probability calibration and a review of who gets flagged and who gets missed, by subgroup.
- At the tuned threshold, about 29 percent of encounters are flagged and roughly four of five flags are false alarms. That volume only makes sense as a screening prompt for a human, never as an automatic decision.
- Two race categories show up in the top permutation importances. That is a property of this dataset and its recording practices, and it is enough on its own to keep this model out of any real workflow.
- Patients discharged to hospice or who died during the encounter are excluded from training and refused at predict time, so the model has nothing to say about them.
- Not for clinical use. The intended use is written into the model card and it means exactly what it says.

## Drift check

src/readmission/drift.py computes the population stability index (PSI) for the numeric features against artifacts/reference_sample.json, a 2,000-row sample of the training data. The reading convention is the usual one: under 0.1 is stable, 0.1 to 0.25 is a shift worth watching, above 0.25 is significant. In a local run, the reference sample compared against itself scores 0.0 on all ten numeric features, and doubling num_medications in the incoming batch pushes that feature to PSI 1.63, flagged significant. There is no drift endpoint in the API; the module is meant to run inside a batch job that scores a fresh extract and pages a human when a feature moves.

## Project layout

```text
src/readmission/
  api.py        FastAPI app: /healthz, /model-info, /predict, /predict/batch
  features.py   Cleaning and feature engineering shared by training and serving
  train.py      Model comparison, threshold tuning, artifact writing
  drift.py      PSI drift checks against the reference sample
tests/          API, features, training, and drift tests, plus a 360-row
                real-data sample under tests/fixtures/
artifacts/      model.joblib, metrics.json, reference_sample.json
docs/adr/       Architecture decision records (model choice, threshold)
data/           README with dataset source and download steps; the CSV
                itself is not committed
Dockerfile, render.yaml, Makefile, pyproject.toml
```
