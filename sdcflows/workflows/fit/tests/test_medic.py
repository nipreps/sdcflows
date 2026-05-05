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

from ..medic import INPUT_FIELDS, _unpack_metadata, init_medic_wf


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
        'magmrg',
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


def test_unpack_metadata_rejects_mixed_pe():
    metadata = [
        {'EchoTime': 0.0142, 'TotalReadoutTime': 0.5, 'PhaseEncodingDirection': 'j'},
        {'EchoTime': 0.03893, 'TotalReadoutTime': 0.5, 'PhaseEncodingDirection': 'j-'},
    ]
    with pytest.raises(ValueError, match='PhaseEncodingDirection'):
        _unpack_metadata(metadata)


@pytest.mark.slow
def test_medic_run(tmpdir, datadir, workdir, outdir):
    """End-to-end MEDIC run on a real multi-echo BOLD.

    Skipped if ``warpkit`` is unavailable (default CI install) or if the
    expected ``ds005250`` multi-echo dataset is not present in the test data
    fixture directory. Use ``TEST_DATA_HOME`` and install
    ``sdcflows[warpkit]`` locally to opt in.
    """
    pytest.importorskip('warpkit')

    pattern = 'ds005250/sub-04/ses-2/func/*_part-mag_bold.nii.gz'
    magnitude_files = sorted(Path(datadir).glob(pattern))
    if not magnitude_files:
        pytest.skip(f'no MEDIC fixtures found under {datadir}/ds005250')

    phase_files = [f.with_name(f.name.replace('part-mag', 'part-phase')) for f in magnitude_files]
    metadata = [
        loads(f.with_name(f.name.replace('.nii.gz', '.json')).read_text()) for f in phase_files
    ]

    tmpdir.chdir()
    medic_wf = init_medic_wf(omp_nthreads=2, debug=True)
    medic_wf.inputs.inputnode.magnitude = [str(f) for f in magnitude_files]
    medic_wf.inputs.inputnode.phase = [str(f) for f in phase_files]
    medic_wf.inputs.inputnode.metadata = metadata

    if workdir:
        medic_wf.base_dir = str(workdir)
    medic_wf.run(plugin='Linear')
