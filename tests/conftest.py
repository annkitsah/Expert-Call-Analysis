from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import ROOT, Settings
from app.main import create_app
from app.models import Corpus
from app.store import load_sample
from tests.fakes import FakeLLM

SAMPLE_DIR = ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return load_sample(SAMPLE_DIR)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    # Explicit values so a developer's real .env / environment can never leak into tests.
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        groq_api_key=None,
        sample_dir=SAMPLE_DIR,
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def client(settings: Settings, fake_llm: FakeLLM) -> TestClient:
    return TestClient(create_app(settings, llm=fake_llm))
