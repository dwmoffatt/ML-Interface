import pytest
from src.MLInterface import MLInterface


@pytest.fixture
def ml_interface():
    ml_interface = MLInterface()

    yield ml_interface
