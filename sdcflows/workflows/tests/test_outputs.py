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
"""Construction-only tests for :mod:`sdcflows.workflows.outputs`."""

from ..outputs import init_fmap_derivatives_wf


def test_fmap_derivatives_wf_default(tmp_path):
    """Default workflow should expose only the static fieldmap sinks."""
    wf = init_fmap_derivatives_wf(output_dir=str(tmp_path))
    assert wf.get_node('ds_fieldmap') is not None
    assert wf.get_node('ds_reference') is not None
    # write_mask off by default.
    assert wf.get_node('ds_mask') is None


def test_fmap_derivatives_wf_merge_fmap_allows_4d(tmp_path):
    """``merge_fmap`` must accept 4D inputs so MEDIC's per-frame Hz fmap flows
    through the same sink as the static estimators' 3D fmaps."""
    wf = init_fmap_derivatives_wf(output_dir=str(tmp_path))
    merge_fmap = wf.get_node('merge_fmap')
    assert merge_fmap is not None
    assert merge_fmap.inputs.allow_4D is True
