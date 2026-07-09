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
"""Test pepolar type of fieldmaps."""

from pathlib import Path

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline import engine as pe

from ..pepolar import _select_topup_config, init_topup_wf


@pytest.mark.parametrize(
    'topup_config',
    [
        '/path/to/custom.cnf',
        Path('/path/to/custom.cnf'),
    ],
)
def test_topup_config(topup_config):
    """Ensure a custom topup config is passed to the node as a str, not a Path."""
    wf = init_topup_wf(topup_config=topup_config)
    assert isinstance(wf.get_node('topup').inputs.config, str), 'topup config must be str'


@pytest.mark.parametrize(
    ('shape', 'sloppy', 'expected'),
    [
        ((50, 50, 50), False, 'b02b0_2.cnf'),
        ((51, 50, 50), False, 'b02b0_1.cnf'),
        ((48, 48, 48), False, 'b02b0_4.cnf'),
        ((50, 50, 50), True, 'b02b0_2_quick.cnf'),
        ((51, 50, 50), True, 'b02b0_1_quick.cnf'),
        ((48, 48, 48), True, 'b02b0_4_quick.cnf'),
    ],
)
def test_select_topup_config(tmp_path, shape, sloppy, expected):
    """The config selector picks an appropriate (optionally quick) config."""
    in_file = tmp_path / 'epi.nii.gz'
    nb.Nifti1Image(np.zeros(shape, dtype='float32'), np.eye(4)).to_filename(in_file)

    selected = _select_topup_config(str(in_file), sloppy=sloppy)

    assert Path(selected).name == expected
    assert Path(selected).exists()


@pytest.mark.slow
@pytest.mark.parametrize('ds', ('ds001771', 'HCP101006'))
def test_topup_wf(tmpdir, bids_layouts, workdir, outdir, ds):
    """Test preparation workflow."""
    layout = bids_layouts[ds]
    epi_path = sorted(
        layout.get(suffix='epi', extension=['nii', 'nii.gz'], scope='raw'),
        key=lambda k: k.path,
    )
    in_data = [f.path for f in epi_path]

    wf = pe.Workflow(name=f'topup_{ds}')
    topup_wf = init_topup_wf(omp_nthreads=2, debug=True, sloppy=True)
    metadata = [layout.get_metadata(f.path) for f in epi_path]

    topup_wf.inputs.inputnode.in_data = in_data
    topup_wf.inputs.inputnode.metadata = metadata

    if outdir:
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        outdir = outdir / 'unittests' / f'topup_{ds}'
        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(outdir),
            write_coeff=True,
            bids_fmap_id='pepolar_id',
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = in_data
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = metadata

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(outdir),
            fmap_type='pepolar',
        )
        fmap_reports_wf.inputs.inputnode.source_files = in_data

        # fmt: off
        wf.connect([
            (topup_wf, fmap_reports_wf, [("outputnode.fmap", "inputnode.fieldmap"),
                                         ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                                         ("outputnode.fmap_mask", "inputnode.fmap_mask")]),
            (topup_wf, fmap_derivatives_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
            ]),
        ])
        # fmt: on
    else:
        wf.add_nodes([topup_wf])

    if workdir:
        wf.base_dir = str(workdir)

    wf.run(plugin='Linear')
