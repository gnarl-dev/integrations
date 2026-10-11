"""Integration tests run against a real node at $GNARL_URL and skip without one."""

from __future__ import annotations

import os
import uuid

import pytest
from gnarl import Client


@pytest.fixture(scope="session")
def client() -> Client:
    if not os.environ.get("GNARL_URL"):
        pytest.skip("GNARL_URL is not set")
    c = Client()
    try:
        c.ping()
    except Exception as e:  # noqa: BLE001 - any failure means "no node here"
        pytest.skip(f"no Gnarl node at {os.environ['GNARL_URL']}: {e}")
    return c


@pytest.fixture
def run_id() -> str:
    """Unique per test, so reruns against one node never see earlier data."""
    return uuid.uuid4().hex[:10]
