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

from ..dynamic import INPUT_FIELDS, init_dynamic_unwarp_wf


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

    for node_name in ('rotime', 'convert_fmap', 'apply_warp', 'average'):
        assert wf.get_node(node_name) is not None, f'missing node {node_name!r}'


# Mirror of ``MEDIC_FIXTURES`` in ``test_medic.py``. Kept duplicated rather than
# imported to avoid cross-test-module coupling; update both lists together.
MEDIC_FIXTURES = [
    pytest.param(
        'ds007637',
        'sub-04/ses-2/func/sub-04_ses-2_task-fracback_acq-MBME_echo-*_part-mag_bold.nii.gz',
        id='ds007637',
    ),
    pytest.param(
        'ds006926',
        'sub-a01/func/sub-a01_task-VisMot_acq-tr1800_echo-*_part-mag_bold.nii.gz',
        id='ds006926',
    ),
]


@pytest.mark.veryslow
@pytest.mark.parametrize(('dataset', 'pattern'), MEDIC_FIXTURES)
def test_dynamic_unwarp_run(tmpdir, datadir, workdir, dataset, pattern):
    """End-to-end run: estimate via MEDIC then apply via this workflow.

    Skipped without ``warpkit`` or without the multi-echo fixture under
    ``$TEST_DATA_HOME``. See ``test_medic_run`` for fetch instructions.
    """
    pytest.importorskip('warpkit')

    from sdcflows.workflows.fit.medic import init_medic_wf

    full_pattern = f'{dataset}/{pattern}'
    magnitude_files = sorted(Path(datadir).glob(full_pattern))
    if not magnitude_files:
        pytest.skip(f'no MEDIC fixtures found under {datadir}/{dataset}')

    phase_files = [f.with_name(f.name.replace('part-mag', 'part-phase')) for f in magnitude_files]
    metadata = [
        loads(f.with_name(f.name.replace('.nii.gz', '.json')).read_text()) for f in phase_files
    ]

    tmpdir.chdir()
    fit_wf = init_medic_wf(omp_nthreads=2)
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
