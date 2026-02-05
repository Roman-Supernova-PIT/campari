import pytest # noqa: F401
import argparse

import tox # noqa: F401
from tox.pytest import init_fixture # noqa: F401

from snappl.config import Config


@pytest.fixture(scope='module')
def cfg():
    return Config.get()

# Source - https://stackoverflow.com/a
# Posted by clay, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-06, License - CC BY-SA 4.0

def pytest_addoption(parser):
    parser.addoption("--overwrite_meta", action=argparse.BooleanOptionalAction, default=False)
    # This option allows you to run tests that check against regression lightcurve files and have them overwrite
    # the metadata. I.e., if the file only differs in metadata, but the fluxes and errors are the same, this option
    # will make those pass.
    # Specifically, if you run with this option, the compare_lightcurve function only checks if the data columns match.
    # If they do, it will overwrite the metadata in the regression file with that of the test file.
    # Then it reruns itself with overwrite_metadata = False to ensure nothing else goes wrong.
    # This is a little scary because it directly modifies the regression files, but it is useful when you have
    # legitimate changes to metadata (e.g., changing provenance info) that would otherwise cause all tests to fail.
    # Use with caution! Of course, worst case scenario, this only changes metdata, and you could check git history
    # to figure out what changed.


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.overwrite_meta
    if "overwrite_meta" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("overwrite_meta", [option_value])

@pytest.fixture(scope="session", autouse=True)
def init_config():
    Config.init("/campari/examples/perlmutter/campari_config_test.yaml", setdefault=True)
