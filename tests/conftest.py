"""Shared fixtures: train a small model on the real-data sample, offline."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from readmission.train import train

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "sample_encounters.csv"


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("artifacts")
    train(FIXTURE_CSV, target)
    return target


@pytest.fixture(scope="session")
def client(artifacts_dir: Path):
    from fastapi.testclient import TestClient

    from readmission import api

    os.environ["ARTIFACTS_DIR"] = str(artifacts_dir)
    api._store.load()
    with TestClient(api.app) as test_client:
        yield test_client
