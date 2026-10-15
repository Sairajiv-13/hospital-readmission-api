VALID_ENCOUNTER = {
    "age": "[70-80)",
    "race": "Caucasian",
    "gender": "Female",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 5,
    "num_lab_procedures": 41,
    "num_procedures": 1,
    "num_medications": 12,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 0,
    "number_diagnoses": 8,
    "diag_1": "414.01",
    "diag_2": "250.0",
    "diag_3": "401.9",
    "change": "Ch",
    "diabetesMed": "Yes",
    "medications": {"insulin": "Steady", "metformin": "Steady"},
}


def test_healthz(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "1.0.0"
    assert "roc_auc" in body["metrics"]
    assert "age_midpoint" in body["features"]


def test_predict_shape(client) -> None:
    response = client.post("/predict", json=VALID_ENCOUNTER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["readmission_probability"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "high"}
    assert body["model_version"] == "1.0.0"
    assert isinstance(body["top_contributing_features"], list)


def test_predict_high_risk_flag_uses_card_threshold(client) -> None:
    info = client.get("/model-info").json()
    threshold = info["decision_threshold"]
    assert 0.0 < threshold < 1.0
    assert info["threshold_analysis"]["tuned_threshold"] == threshold

    body = client.post("/predict", json=VALID_ENCOUNTER).json()
    assert isinstance(body["high_risk"], bool)
    assert body["high_risk"] == (body["readmission_probability"] >= threshold)


def test_predict_validation_error_is_clean_422(client) -> None:
    bad = dict(VALID_ENCOUNTER, age="seventy", time_in_hospital=99)
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_unknown_medication_value_is_422(client) -> None:
    bad = dict(VALID_ENCOUNTER, medications={"insulin": "Lots"})
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_out_of_scope_discharge_is_422(client) -> None:
    bad = dict(VALID_ENCOUNTER, discharge_disposition_id=11)
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_batch_predict(client) -> None:
    payload = {"encounters": [VALID_ENCOUNTER, dict(VALID_ENCOUNTER, age="[50-60)")]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert all(0.0 <= item["readmission_probability"] <= 1.0 for item in body)


def test_batch_cap_is_enforced(client) -> None:
    payload = {"encounters": [VALID_ENCOUNTER] * 101}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_batch_empty_is_422(client) -> None:
    response = client.post("/predict/batch", json={"encounters": []})
    assert response.status_code == 422
