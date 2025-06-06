import pathlib
import pytest # noqa: F401

import tox # noqa: F401
from tox.pytest import init_fixture # noqa: F401

from snpit_utils.config import Config


@pytest.fixture(scope='session')
def config_path():
    return pathlib.Path(__file__).parent.parent.parent / 'examples/sample_campari_config.yaml'


@pytest.fixture(scope='module')
def cfg(config_path):
    return Config.get(config_path, setdefault=True)
