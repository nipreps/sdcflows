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
"""Tests for the per-volume dynamic apply workflow."""

from json import loads
from pathlib import Path

import pytest

from ..dynamic import INPUT_FIELDS, init_dynamic_unwarp_wf

# See test_medic._MEDIC_TEST_VOLUMES — keep these in sync.
_MEDIC_TEST_VOLUMES = 3


def _truncate_to_volumes(in_files, volumes, dest):
    import nibabel as nb

    out = []
    for f in in_files:
        img = nb.load(str(f))
        if img.shape[-1] > volumes:
            img = img.slicer[..., :volumes]
        new = dest / f.name
        img.to_filename(new)
        out.append(new)
    return out


def test_dynamic_unwarp_construct():
    """Build the workflow and verify shape."""
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
    }

    for node_name in ('rotime', 'unwarp', 'average'):
        assert wf.get_node(node_name) is not None, f'missing node {node_name!r}'


def test_dynamic_unwarp_jacobian_flag_propagates():
    """The ``jacobian`` ctor flag forwards to the per-volume resampler."""
    wf = init_dynamic_unwarp_wf(jacobian=False)
    unwarp = wf.get_node('unwarp')
    assert unwarp.inputs.jacobian is False

    wf = init_dynamic_unwarp_wf(jacobian=True)
    unwarp = wf.get_node('unwarp')
    assert unwarp.inputs.jacobian is True


def test_apply_dynamic_unwarp_matches_static(tmp_path, monkeypatch):
    """For a 4D fmap with identical frames, per-volume resampling matches the
    static path frame-by-frame.

    This pins :func:`sdcflows.transform.apply_dynamic_unwarp` to the same
    Hz→VSM + scipy.ndimage convention as the rest of the codebase — if the
    static path ever changes its sign or pe_info handling, this test catches
    the drift.
    """
    import nibabel as nb
    import numpy as np

    from sdcflows.transform import _sdc_unwarp, apply_dynamic_unwarp
    from sdcflows.utils.tools import ensure_positive_cosines

    monkeypatch.chdir(tmp_path)

    rng = np.random.default_rng(0)
    shape = (5, 7, 5)
    n_frames = 3
    affine = np.eye(4)

    fmap_3d = rng.normal(scale=0.5, size=shape).astype('float32')
    fmap_4d = np.broadcast_to(fmap_3d[..., None], (*shape, n_frames)).astype('float32')
    distorted = rng.normal(size=(*shape, n_frames)).astype('float32')

    distorted_path = tmp_path / 'distorted.nii.gz'
    fmap_path = tmp_path / 'fmap.nii.gz'
    nb.Nifti1Image(distorted, affine).to_filename(distorted_path)
    nb.Nifti1Image(fmap_4d, affine).to_filename(fmap_path)

    resampled = apply_dynamic_unwarp(
        str(distorted_path),
        str(fmap_path),
        pe_dir='j',
        ro_time=0.1,
        jacobian=True,
        order=1,
        prefilter=False,
        num_threads=1,
        allow_negative=True,
    )
    out_data = np.asanyarray(resampled.dataobj)

    # Run the same primitive directly, per-frame, with no parallelism.
    img, axcodes = ensure_positive_cosines(nb.load(str(distorted_path)))
    voxcoords = np.indices(shape, dtype='float32')
    pe_axis = 'ijk'.index('j')
    flip = (axcodes[pe_axis] in 'LPI') ^ False
    pe_info = (pe_axis, -0.1 if flip else 0.1)
    expected = np.stack(
        [
            _sdc_unwarp(
                distorted[..., t],
                voxcoords.copy(),
                pe_info,
                None,
                jacobian=True,
                fmap_hz=fmap_3d,
                output_dtype='float32',
                order=1,
                prefilter=False,
            )
            for t in range(n_frames)
        ],
        axis=-1,
    )
    assert np.allclose(out_data, expected, atol=1e-5)


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
    trunc_dir = Path(str(tmpdir)) / 'trunc'
    trunc_dir.mkdir(exist_ok=True)
    magnitude_files = _truncate_to_volumes(magnitude_files, _MEDIC_TEST_VOLUMES, trunc_dir)
    phase_files = _truncate_to_volumes(phase_files, _MEDIC_TEST_VOLUMES, trunc_dir)

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
    wf.connect([(fit_wf, apply_wf, [('outputnode.fmap', 'inputnode.fmap')])])

    if workdir:
        wf.base_dir = str(workdir)
    wf.run(plugin='Linear')
