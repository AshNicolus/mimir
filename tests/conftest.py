import pytest

from mimir import Mimir


@pytest.fixture
def memory():
    m = Mimir(db_path=":memory:")
    yield m
    m.close()
