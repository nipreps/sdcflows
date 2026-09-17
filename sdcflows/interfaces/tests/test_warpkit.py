# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""Tests for the warpkit nipype interface wrappers.

These tests check spec shape and small helpers; the actual ``_run_interface``
methods require :mod:`warpkit`, which is an optional dependency.
"""

import pytest

from sdcflows.interfaces import warpkit as wk


def test_warpkit_base_interface_pkg():
    """All warpkit interfaces share the ``warpkit`` library tag.

    This lets nipype's ``LibraryBaseInterface`` emit a single, consistent
    "warpkit not installed" message rather than per-class noise.
    """
    assert wk.WarpkitBaseInterface._pkg == 'warpkit'
    for cls in (wk.UnwrapPhase, wk.ComputeFieldmap):
        assert issubclass(cls, wk.WarpkitBaseInterface)


@pytest.mark.parametrize(
    'cls,expected_inputs,expected_outputs',
    [
        (
            wk.UnwrapPhase,
            {'phase', 'magnitude', 'echo_times'},
            {'unwrapped', 'masks'},
        ),
        (
            wk.ComputeFieldmap,
            {'unwrapped', 'magnitude', 'masks', 'border_filt', 'svd_filt'},
            {'fieldmap_native', 'displacement_map', 'fieldmap'},
        ),
    ],
)
def test_interface_spec_traits(cls, expected_inputs, expected_outputs):
    """Each interface declares the expected input/output traits."""
    iface = cls()
    assert expected_inputs <= set(iface.inputs.copyable_trait_names())
    assert expected_outputs <= set(iface.output_spec().copyable_trait_names())


def test_compute_fieldmap_border_filt_default():
    """``border_filt`` defaults to ``(1, 5)``.

    Regression test for an upstream ``traits.Tuple`` quirk where the outer
    ``default`` kwarg silently lost to inner ``Int()`` zeros, collapsing the
    SVD border filter and clipping the dynamic fieldmap footprint.
    """
    iface = wk.ComputeFieldmap()
    assert tuple(iface.inputs.border_filt) == (1, 5)
