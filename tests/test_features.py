from pathlib import Path

import pandas as pd

from readmission.features import age_to_midpoint, clean_encounters, group_diag

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_encounters.csv"


def test_age_to_midpoint() -> None:
    assert age_to_midpoint("[70-80)") == 75.0
    assert age_to_midpoint("[0-10)") == 5.0


def test_group_diag_chapters() -> None:
    assert group_diag("250.83") == "Diabetes"
    assert group_diag("404.1") == "Circulatory"
    assert group_diag("486") == "Respiratory"
    assert group_diag("V57") == "Other"
    assert group_diag("?") == "Missing"


def test_clean_encounters_drops_hospice_and_death() -> None:
    raw = pd.read_csv(FIXTURE_CSV, low_memory=False)
    cleaned = clean_encounters(raw)
    assert len(cleaned) < len(raw)
    assert not cleaned["discharge_disposition_id"].isin(["11", "13", "14", "19", "20", "21"]).any()
    assert set(cleaned["readmitted_30d"].unique()) <= {0, 1}
    assert cleaned["readmitted_30d"].sum() > 0
