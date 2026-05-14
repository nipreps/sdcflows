# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2025 The NiPreps Developers <nipreps@gmail.com>
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
    # write_dynamic / write_mask off by default.
    assert wf.get_node('ds_mask') is None
    assert wf.get_node('ds_fmap_dynamic') is None
    assert wf.get_node('ds_fmap_dynamic_ref') is None
    assert wf.get_node('ds_fmap_dynamic_mask') is None


def test_fmap_derivatives_wf_write_dynamic(tmp_path):
    """``write_dynamic=True`` adds the 4D MEDIC sinks and tags the fieldmap."""
    wf = init_fmap_derivatives_wf(
        output_dir=str(tmp_path),
        write_dynamic=True,
        bids_fmap_id='medic_0',
    )

    ds_fmap_dynamic = wf.get_node('ds_fmap_dynamic')
    ds_fmap_dynamic_ref = wf.get_node('ds_fmap_dynamic_ref')
    ds_fmap_dynamic_mask = wf.get_node('ds_fmap_dynamic_mask')
    assert ds_fmap_dynamic is not None
    assert ds_fmap_dynamic_ref is not None
    assert ds_fmap_dynamic_mask is not None

    # The Hz dynamic sink carries Units + B0FieldIdentifier so downstream
    # consumers can join it to the same B0 group as the static fieldmap.
    assert ds_fmap_dynamic.inputs.Units == 'Hz'
    assert ds_fmap_dynamic.inputs.B0FieldIdentifier == 'medic_0'

    # `desc` distinguishes the three 4D outputs at the BIDS-path level —
    # niworkflows' nipreps.json only has fmap path patterns for the
    # fieldmap/mask suffixes, so the magnitude ref reuses the fieldmap suffix.
    assert ds_fmap_dynamic.inputs.desc == 'dynamic'
    assert ds_fmap_dynamic.inputs.suffix == 'fieldmap'
    assert ds_fmap_dynamic_ref.inputs.desc == 'dynamicref'
    assert ds_fmap_dynamic_ref.inputs.suffix == 'fieldmap'
    assert ds_fmap_dynamic_mask.inputs.desc == 'dynamicbrain'
    assert ds_fmap_dynamic_mask.inputs.suffix == 'mask'

    # Each dynamic sink should land under the fmap/ datatype.
    for node in (ds_fmap_dynamic, ds_fmap_dynamic_ref, ds_fmap_dynamic_mask):
        assert node.inputs.datatype == 'fmap'


def test_fmap_derivatives_wf_write_dynamic_no_b0id(tmp_path):
    """Without ``bids_fmap_id``, the Hz dynamic sink omits B0FieldIdentifier."""
    wf = init_fmap_derivatives_wf(output_dir=str(tmp_path), write_dynamic=True)
    ds_fmap_dynamic = wf.get_node('ds_fmap_dynamic')
    # ``B0FieldIdentifier`` is set as a dynamic trait only when bids_fmap_id is
    # provided; without it the attribute should not exist on the sink inputs.
    assert not hasattr(ds_fmap_dynamic.inputs, 'B0FieldIdentifier')
