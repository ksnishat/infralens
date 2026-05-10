"""Shared pytest fixtures for all InfraLens tests."""

import os
import tempfile
import pytest


@pytest.fixture(scope="session")
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def env_vars(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_LOG_LEVEL", "WARNING")