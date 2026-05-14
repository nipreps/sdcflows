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
"""Tests for the MEDIC dynamic fieldmap workflow."""

from json import loads
from pathlib import Path

import pytest

from ..medic import INPUT_FIELDS, _first, _temporal_mean, _unpack_metadata, init_medic_wf

# A handful of timepoints is enough to exercise the full per-volume MEDIC
# path; the source datasets ship 200+ volumes × 5 echoes × mag+phase, which
# OOM-kills CI runners when xdist schedules these fixtures in parallel.
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


def test_medic_construct():
    """Build the workflow and verify its surface — no warpkit required.

    This guards against module-load regressions and confirms the inputnode/
    outputnode shape that ``init_fmap_preproc_wf`` depends on.
    """
    wf = init_medic_wf()
    assert wf.name == 'medic_wf'

    inputnode = wf.get_node('inputnode')
    outputnode = wf.get_node('outputnode')
    assert inputnode is not None and outputnode is not None
    assert set(inputnode.outputs.copyable_trait_names()) >= set(INPUT_FIELDS)
    assert set(outputnode.inputs.copyable_trait_names()) >= {
        'fmap',
        'fmap_dynamic',
        'fmap_ref',
        'fmap_dynamic_ref',
        'fmap_mask',
        'fmap_dynamic_mask',
        'fmap_coeff',
        'method',
    }
    # method is set unconditionally at construction.
    assert outputnode.inputs.method.startswith('MEDIC')

    # Core nodes wired in the order the workflow describes.
    for name in (
        'extract_meta',
        'unwrap',
        'compute_fmap',
        'fmap_mean',
        'pick_mag1',
        'magnitude_wf',
        'bs_filter',
    ):
        assert wf.get_node(name) is not None, f'missing node {name!r}'


def test_unpack_metadata_converts_te_to_ms():
    metadata = [
        {'EchoTime': 0.0142, 'TotalReadoutTime': 0.5, 'PhaseEncodingDirection': 'j'},
        {'EchoTime': 0.03893, 'TotalReadoutTime': 0.5, 'PhaseEncodingDirection': 'j'},
    ]
    tes, trt, ped = _unpack_metadata(metadata)
    assert tes == [pytest.approx(14.2), pytest.approx(38.93)]
    assert trt == 0.5
    assert ped == 'j'


def test_unpack_metadata_rejects_single_echo():
    metadata = [{'EchoTime': 0.0142, 'TotalReadoutTime': 0.5, 'PhaseEncodingDirection': 'j'}]
    with pytest.raises(ValueError, match='at least two echoes'):
        _unpack_metadata(metadata)


def test_unpack_metadata_rejects_mixed_pe():
    metadata = [
        {'EchoTime': 0.0142, 'TotalReadoutTime': 0.5, 'PhaseEncodingDirection': 'j'},
        {'EchoTime': 0.03893, 'TotalReadoutTime': 0.5, 'PhaseEncodingDirection': 'j-'},
    ]
    with pytest.raises(ValueError, match='PhaseEncodingDirection'):
        _unpack_metadata(metadata)


def test_unpack_metadata_rejects_empty():
    with pytest.raises(ValueError, match='per-echo metadata'):
        _unpack_metadata([])


def test_medic_wf_sloppy_sets_zooms_min():
    """``sloppy=True`` lowers the B-spline ``zooms_min`` for the static path."""
    wf = init_medic_wf(sloppy=True)
    bs_filter = wf.get_node('bs_filter')
    assert bs_filter.inputs.zooms_min == 4.0


def test_first_helper():
    """``_first`` returns the head of the list or ``None`` when empty."""
    assert _first(['a', 'b', 'c']) == 'a'
    assert _first([]) is None


def test_temporal_mean_collapses_4d(tmp_path, monkeypatch):
    """``_temporal_mean`` averages along the time axis and emits a 3D NIfTI."""
    import nibabel as nb
    import numpy as np

    monkeypatch.chdir(tmp_path)

    data = np.stack(
        [np.ones((3, 4, 2), dtype='float32') * scalar for scalar in (1.0, 3.0, 5.0)],
        axis=-1,
    )
    in_file = tmp_path / 'in.nii.gz'
    nb.Nifti1Image(data, np.eye(4)).to_filename(in_file)

    out_file = _temporal_mean(str(in_file))
    out_img = nb.load(out_file)
    assert out_img.ndim == 3
    assert np.allclose(np.asanyarray(out_img.dataobj), 3.0)


def test_temporal_mean_passthrough_3d(tmp_path, monkeypatch):
    """3D inputs are written back unchanged (no axis to average over)."""
    import nibabel as nb
    import numpy as np

    monkeypatch.chdir(tmp_path)
    data = np.full((3, 4, 2), 7.0, dtype='float32')
    in_file = tmp_path / 'in3d.nii.gz'
    nb.Nifti1Image(data, np.eye(4)).to_filename(in_file)

    out_file = _temporal_mean(str(in_file))
    assert np.allclose(np.asanyarray(nb.load(out_file).dataobj), 7.0)


# Each entry is (dataset, mag_glob_under_dataset).
# Add new fixtures here — the test self-skips when a dataset isn't on disk.
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
def test_medic_run(tmpdir, datadir, workdir, outdir, dataset, pattern):
    """End-to-end MEDIC run on a real multi-echo BOLD.

    Skipped if ``warpkit`` is unavailable (default CI install) or if the
    expected dataset is not present under ``$TEST_DATA_HOME``. To opt in,
    install ``sdcflows[warpkit]`` and stage the dataset, e.g.

    .. code-block:: console

        # ds006926: OpenNeuro multi-echo mag+phase BOLD (publicly available)
        cd $TEST_DATA_HOME
        datalad install https://github.com/OpenNeuroDatasets/ds006926.git
        datalad get -d ds006926 sub-a01/func/sub-a01_task-VisMot_acq-tr1800_*

    """
    pytest.importorskip('warpkit')

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

    medic_wf = init_medic_wf(omp_nthreads=2)
    medic_wf.inputs.inputnode.magnitude = [str(f) for f in magnitude_files]
    medic_wf.inputs.inputnode.phase = [str(f) for f in phase_files]
    medic_wf.inputs.inputnode.metadata = metadata

    if workdir:
        medic_wf.base_dir = str(workdir)
    medic_wf.run(plugin='Linear')
