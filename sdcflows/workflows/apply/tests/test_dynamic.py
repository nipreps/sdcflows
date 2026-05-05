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
"""Tests for the per-volume MEDIC apply workflow."""

from json import loads
from pathlib import Path

import pytest

from ..dynamic import INPUT_FIELDS, _pe_axis, init_dynamic_unwarp_wf


def test_dynamic_unwarp_construct():
    """Build the workflow without warpkit installed and verify shape."""
    wf = init_dynamic_unwarp_wf()
    assert wf.name == 'dynamic_unwarp_wf'

    inputnode = wf.get_node('inputnode')
    outputnode = wf.get_node('outputnode')
    assert inputnode is not None and outputnode is not None
    assert set(inputnode.outputs.copyable_trait_names()) >= set(INPUT_FIELDS)
    assert set(outputnode.inputs.copyable_trait_names()) >= {
        'corrected',
        'corrected_ref',
        'corrected_mask',
        'fieldwarp',
    }

    for node_name in ('rotime', 'pe_axis', 'convert_fmap', 'apply_warp', 'average'):
        assert wf.get_node(node_name) is not None, f'missing node {node_name!r}'


@pytest.mark.parametrize(
    'pe_direction,expected',
    [
        ('i', ('i', False)),
        ('i-', ('i', True)),
        ('j', ('j', False)),
        ('j-', ('j', True)),
        ('k', ('k', False)),
        ('k-', ('k', True)),
    ],
)
def test_pe_axis_splits_axis_and_sign(pe_direction, expected):
    assert _pe_axis(pe_direction) == expected


@pytest.mark.slow
def test_dynamic_unwarp_run(tmpdir, datadir, workdir):
    """End-to-end run: estimate via MEDIC then apply via this workflow.

    Skipped without ``warpkit`` or without the multi-echo fixture.
    """
    pytest.importorskip('warpkit')

    from sdcflows.workflows.fit.medic import init_medic_wf

    pattern = 'ds005250/sub-04/ses-2/func/*_part-mag_bold.nii.gz'
    magnitude_files = sorted(Path(datadir).glob(pattern))
    if not magnitude_files:
        pytest.skip(f'no MEDIC fixtures found under {datadir}/ds005250')

    phase_files = [f.with_name(f.name.replace('part-mag', 'part-phase')) for f in magnitude_files]
    metadata = [
        loads(f.with_name(f.name.replace('.nii.gz', '.json')).read_text()) for f in phase_files
    ]

    tmpdir.chdir()
    fit_wf = init_medic_wf(omp_nthreads=2, debug=True)
    fit_wf.inputs.inputnode.magnitude = [str(f) for f in magnitude_files]
    fit_wf.inputs.inputnode.phase = [str(f) for f in phase_files]
    fit_wf.inputs.inputnode.metadata = metadata

    apply_wf = init_dynamic_unwarp_wf(omp_nthreads=2)
    # Use the first-echo magnitude as the distorted target.
    apply_wf.inputs.inputnode.distorted = str(magnitude_files[0])
    apply_wf.inputs.inputnode.metadata = metadata[0]

    from niworkflows.engine.workflows import LiterateWorkflow as Workflow

    wf = Workflow(name=f'medic_apply_{magnitude_files[0].stem.replace(".nii", "")}')
    wf.connect([(fit_wf, apply_wf, [('outputnode.fmap_dynamic', 'inputnode.fmap_dynamic')])])

    if workdir:
        wf.base_dir = str(workdir)
    wf.run(plugin='Linear')
