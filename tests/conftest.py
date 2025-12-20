"""Pytest configuration."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
