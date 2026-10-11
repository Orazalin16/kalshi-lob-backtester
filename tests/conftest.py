import pytest

import klob


@pytest.fixture(scope="session")
def core():
    if not klob.HAVE_CORE:
        pytest.skip("C++ extension not built")
    return klob
