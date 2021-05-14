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
"""Test _version.py."""
import sys
from collections import namedtuple
from pkg_resources import DistributionNotFound
from importlib import reload
import sdcflows


def test_version_scm0(monkeypatch):
    """Retrieve the version via setuptools_scm."""

    class _version:
        __version__ = "10.0.0"

    monkeypatch.setitem(sys.modules, "sdcflows._version", _version)
    reload(sdcflows)
    assert sdcflows.__version__ == "10.0.0"


def test_version_scm1(monkeypatch):
    """Retrieve the version via pkg_resources."""
    monkeypatch.setitem(sys.modules, "sdcflows._version", None)

    def _dist(name):
        Distribution = namedtuple("Distribution", ["name", "version"])
        return Distribution(name, "success")

    monkeypatch.setattr("pkg_resources.get_distribution", _dist)
    reload(sdcflows)
    assert sdcflows.__version__ == "success"


def test_version_scm2(monkeypatch):
    """Check version could not be interpolated."""
    monkeypatch.setitem(sys.modules, "sdcflows._version", None)

    def _raise(name):
        raise DistributionNotFound("No get_distribution mock")

    monkeypatch.setattr("pkg_resources.get_distribution", _raise)
    reload(sdcflows)
    assert sdcflows.__version__ == "unknown"
