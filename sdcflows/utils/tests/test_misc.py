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
"""Test miscellaneous utilities."""

import sys
import types
from collections import namedtuple

import pytest

from ..misc import get_free_mem


@pytest.mark.parametrize('retval', [None, 10])
def test_get_free_mem(monkeypatch, retval):
    """Test the get_free_mem utility."""

    def mock_func():
        if retval is None:
            raise ImportError
        return namedtuple('Mem', ('free',))(free=retval)

    psutil = types.ModuleType('psutil')
    psutil.virtual_memory = mock_func
    monkeypatch.setitem(sys.modules, 'psutil', psutil)
    assert get_free_mem() == retval
