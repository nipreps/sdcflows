"""py.test configuration."""
import os
from pathlib import Path
import numpy
import nibabel
import pytest
from bids.layout import BIDSLayout

test_data_env = os.getenv("TEST_DATA_HOME", str(Path.home() / "sdcflows-tests"))
test_output_dir = os.getenv("TEST_OUTPUT_DIR")
test_workdir = os.getenv("TEST_WORK_DIR")

layouts = {
    p.name: BIDSLayout(str(p), validate=False, derivatives=True)
    for p in Path(test_data_env).glob("*")
    if p.is_dir()
}

data_dir = Path(__file__).parent / "tests" / "data"

layouts["dsA"] = BIDSLayout(data_dir / "dsA", validate=False, derivatives=False)


def pytest_report_header(config):
    return f"""\
TEST_DATA_HOME={test_data_env}
-> Available datasets: {', '.join(layouts.keys())}.
TEST_OUTPUT_DIR={test_output_dir or '<unset> (output files will be discarded)'}.
TEST_WORK_DIR={test_workdir or '<unset> (intermediate files will be discarded)'}.
"""


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["np"] = numpy
    doctest_namespace["nb"] = nibabel
    doctest_namespace["os"] = os
    doctest_namespace["Path"] = Path
    doctest_namespace["layouts"] = layouts
    for key, val in list(layouts.items()):
        doctest_namespace[key] = Path(val.root)

    doctest_namespace["dsA_dir"] = data_dir / "dsA"


@pytest.fixture
def workdir():
    return None if test_workdir is None else Path(test_workdir)


@pytest.fixture
def outdir():
    return None if test_output_dir is None else Path(test_output_dir)


@pytest.fixture
def bids_layouts():
    if layouts:
        return layouts
    pytest.skip()


@pytest.fixture
def datadir():
    return Path(test_data_env)


@pytest.fixture
def testdata_dir():
    return data_dir


@pytest.fixture
def dsA_dir():
    return data_dir / "dsA"
