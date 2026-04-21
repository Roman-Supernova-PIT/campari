import pytest  # noqa: F401

import tox  # noqa: F401
from tox.pytest import init_fixture  # noqa: F401

from snappl.config import Config


@pytest.fixture(scope="module")
def cfg():
    return Config.get()


@pytest.fixture(scope="session", autouse=True)
def init_config():
    Config.init("/home/campari/examples/perlmutter/campari_test_config.yaml", setdefault=True)

# Source - https://stackoverflow.com/a
# Posted by clay, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-06, License - CC BY-SA 4.0


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.overwrite_meta
    if "overwrite_meta" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("overwrite_meta", [option_value])
