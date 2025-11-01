import pytest # noqa: F401

import tox # noqa: F401
from tox.pytest import init_fixture # noqa: F401

from snappl.config import Config


@pytest.fixture(scope='module')
def cfg():
    return Config.get()
