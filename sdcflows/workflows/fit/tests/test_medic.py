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
"""Tests for the MEDIC dynamic fieldmap workflow."""

from json import loads
from pathlib import Path

import pytest

from ..medic import INPUT_FIELDS, _first, _unpack_metadata, init_medic_wf


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
        'fmap_ref',
        'fmap_mask',
        'method',
    }
    # method is set unconditionally at construction.
    assert outputnode.inputs.method.startswith('MEDIC')

    # Core nodes wired in the order the workflow describes.
    for name in (
        'extract_meta',
        'unwrap',
        'compute_fmap',
        'pick_mag1',
    ):
        assert wf.get_node(name) is not None, f'missing node {name!r}'


def test_medic_passes_threads():
    """``omp_nthreads`` must reach warpkit, not only Nipype's ``n_procs``.

    Nipype's ``Node`` forwards ``n_procs`` into an interface's ``num_threads``.
    """
    wf = init_medic_wf(omp_nthreads=4)
    assert wf.get_node('unwrap').inputs.num_threads == 4
    assert wf.get_node('compute_fmap').inputs.num_threads == 4


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


def test_unpack_metadata_effective_echo_spacing(tmp_path):
    """TRT is resolved with ``get_trt``, so ``EffectiveEchoSpacing`` suffices."""
    import nibabel as nb
    import numpy as np

    in_file = tmp_path / 'phase.nii.gz'
    nb.Nifti1Image(np.zeros((10, 11, 4), dtype='float32'), np.eye(4)).to_filename(in_file)
    metadata = [
        {'EchoTime': 0.0142, 'EffectiveEchoSpacing': 0.001, 'PhaseEncodingDirection': 'j'},
        {'EchoTime': 0.03893, 'EffectiveEchoSpacing': 0.001, 'PhaseEncodingDirection': 'j'},
    ]
    _, trt, _ = _unpack_metadata(metadata, in_files=[str(in_file)] * 2)
    assert trt == pytest.approx(0.010)


def test_unpack_metadata_fallback_trt():
    metadata = [
        {'EchoTime': 0.0142, 'PhaseEncodingDirection': 'j'},
        {'EchoTime': 0.03893, 'PhaseEncodingDirection': 'j'},
    ]
    _, trt, _ = _unpack_metadata(metadata, fallback=0.03)
    assert trt == 0.03


def test_unpack_metadata_rejects_mixed_trt():
    metadata = [
        {'EchoTime': 0.0142, 'TotalReadoutTime': 0.5, 'PhaseEncodingDirection': 'j'},
        {'EchoTime': 0.03893, 'TotalReadoutTime': 0.4, 'PhaseEncodingDirection': 'j'},
    ]
    with pytest.raises(ValueError, match='total readout time'):
        _unpack_metadata(metadata)


def test_first_helper():
    """``_first`` returns the head of the list or ``None`` when empty."""
    assert _first(['a', 'b', 'c']) == 'a'
    assert _first([]) is None


@pytest.mark.veryslow
def test_medic_run(
    tmpdir, datadir, workdir, outdir, medic_fixture, medic_test_volumes, truncate_to_volumes
):
    """End-to-end MEDIC run on a real multi-echo BOLD.

    Skipped if ``warpkit`` is unavailable or if the
    expected dataset is not present under ``$TEST_DATA_HOME``. To run it,
    stage the dataset, e.g.

    .. code-block:: console

        # ds006926: OpenNeuro multi-echo mag+phase BOLD (publicly available)
        cd $TEST_DATA_HOME
        datalad install https://github.com/OpenNeuroDatasets/ds006926.git
        datalad get -d ds006926 sub-a01/func/sub-a01_task-VisMot_acq-tr1800_*

    """
    pytest.importorskip('warpkit')

    dataset, pattern = medic_fixture
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
    magnitude_files = truncate_to_volumes(magnitude_files, medic_test_volumes, trunc_dir)
    phase_files = truncate_to_volumes(phase_files, medic_test_volumes, trunc_dir)

    medic_wf = init_medic_wf(omp_nthreads=2)
    medic_wf.inputs.inputnode.magnitude = [str(f) for f in magnitude_files]
    medic_wf.inputs.inputnode.phase = [str(f) for f in phase_files]
    medic_wf.inputs.inputnode.metadata = metadata

    if workdir:
        medic_wf.base_dir = str(workdir)
    medic_wf.run(plugin='Linear')
