# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""py.test configuration."""

import os
import logging
from pathlib import Path
import numpy
import nibabel
import pytest
from bids.layout import BIDSLayout
from .fieldmaps import clear_registry

# disable ET
os.environ['NO_ET'] = '1'

test_data_env = os.getenv("TEST_DATA_HOME", str(Path.home() / "sdcflows-tests"))
test_output_dir = os.getenv("TEST_OUTPUT_DIR")
test_workdir = os.getenv("TEST_WORK_DIR")
_sloppy_mode = os.getenv("TEST_PRODUCTION", "off").lower() not in ("on", "1", "true", "yes", "y")

layouts = {
    p.name: BIDSLayout(str(p), validate=False, derivatives=True)
    for p in Path(test_data_env).glob("*")
    if p.is_dir()
}

data_dir = Path(__file__).parent / "tests" / "data"
layouts.update({
    folder.name: BIDSLayout(folder, validate=False, derivatives=False)
    for folder in data_dir.glob("ds*") if folder.is_dir()
})


def pytest_report_header(config):
    return f"""\
TEST_DATA_HOME={test_data_env}
-> Available datasets: {', '.join(layouts.keys())}.
TEST_OUTPUT_DIR={test_output_dir or '<unset> (output files will be discarded)'}.
TEST_WORK_DIR={test_workdir or '<unset> (intermediate files will be discarded)'}.
"""


@pytest.fixture(autouse=True)
def doctest_fixture(doctest_namespace, request, caplog):
    doctest_plugin = request.config.pluginmanager.getplugin("doctest")
    if isinstance(request.node, doctest_plugin.DoctestItem):
        doctest_namespace.update(
            np=numpy,
            nb=nibabel,
            os=os,
            Path=Path,
            layouts=layouts,
            dsA_dir=data_dir / "dsA",
            dsB_dir=data_dir / "dsB",
            dsC_dir=data_dir / "dsC",
            data_dir=data_dir,
            caplog=caplog,
            logging=logging,
        )
        doctest_namespace.update((key, Path(val.root)) for key, val in layouts.items())

        # Start every doctest clean, and clean up after ourselves
        clear_registry()
        yield
        clear_registry()
    else:
        yield


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


@pytest.fixture
def sloppy_mode():
    return _sloppy_mode
