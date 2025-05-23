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
"""Test fieldmap-less SDC-SyN."""

import json

import nibabel as nb
import numpy as np
import pytest
from nipype.pipeline import engine as pe

from .... import data
from ..syn import (
    _adjust_zooms,
    _mm2vox,
    _set_dtype,
    _warp_dir,
    init_syn_preprocessing_wf,
    init_syn_sdc_wf,
)


@pytest.mark.veryslow
@pytest.mark.slow
def test_syn_wf(tmpdir, datadir, workdir, outdir, sloppy_mode):
    """Build and run an SDC-SyN workflow."""
    derivs_path = datadir / 'ds000054' / 'derivatives'
    smriprep = derivs_path / 'smriprep-0.6' / 'sub-100185' / 'anat'

    wf = pe.Workflow(name='test_syn')

    prep_wf = init_syn_preprocessing_wf(
        omp_nthreads=4,
        debug=sloppy_mode,
        auto_bold_nss=True,
        t1w_inversion=True,
    )
    prep_wf.inputs.inputnode.in_epis = [
        str(
            datadir
            / 'ds000054'
            / 'sub-100185'
            / 'func'
            / 'sub-100185_task-machinegame_run-01_bold.nii.gz'
        ),
        str(
            datadir
            / 'ds000054'
            / 'sub-100185'
            / 'func'
            / 'sub-100185_task-machinegame_run-02_bold.nii.gz'
        ),
    ]
    prep_wf.inputs.inputnode.in_meta = [
        json.loads((datadir / 'ds000054' / 'task-machinegame_bold.json').read_text()),
    ] * 2
    prep_wf.inputs.inputnode.std2anat_xfm = str(
        smriprep / 'sub-100185_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5'
    )
    prep_wf.inputs.inputnode.in_anat = str(smriprep / 'sub-100185_desc-preproc_T1w.nii.gz')
    prep_wf.inputs.inputnode.mask_anat = str(smriprep / 'sub-100185_desc-brain_mask.nii.gz')

    syn_wf = init_syn_sdc_wf(
        debug=sloppy_mode,
        sloppy=sloppy_mode,
        omp_nthreads=4,
    )

    # fmt: off
    wf.connect([
        (prep_wf, syn_wf, [
            ("outputnode.epi_ref", "inputnode.epi_ref"),
            ("outputnode.epi_mask", "inputnode.epi_mask"),
            ("outputnode.anat_ref", "inputnode.anat_ref"),
            ("outputnode.anat_mask", "inputnode.anat_mask"),
            ("outputnode.sd_prior", "inputnode.sd_prior"),
        ]),
    ])
    # fmt: on

    if outdir:
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        outdir = outdir / 'unittests' / 'test_syn'
        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(outdir),
            write_coeff=True,
            bids_fmap_id='sdcsyn',
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = [
            str(
                derivs_path / 'sdcflows-tests' / 'sub-100185_task-machinegame_run-1_boldref.nii.gz'
            )
        ]
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = {'PhaseEncodingDirection': 'j-'}

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(outdir),
            fmap_type='sdcsyn',
        )
        fmap_reports_wf.inputs.inputnode.source_files = [
            str(
                derivs_path / 'sdcflows-tests' / 'sub-100185_task-machinegame_run-1_boldref.nii.gz'
            )
        ]

        # fmt: off
        wf.connect([
            (syn_wf, fmap_reports_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask")]),
            (syn_wf, fmap_derivatives_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
            ]),
        ])
        # fmt: on

    if workdir:
        wf.base_dir = str(workdir)

    wf.run(plugin='Linear')


@pytest.mark.parametrize('laplacian_weight', [None, (0.5, 0.1), (0.8, -1.0)])
@pytest.mark.parametrize('sloppy', [True, False])
def test_syn_wf_inputs(sloppy, laplacian_weight):
    """Test the input validation of the SDC-SyN workflow."""
    from sdcflows.workflows.fit.syn import MAX_LAPLACIAN_WEIGHT

    laplacian_weight = (
        (0.1, 0.2)
        if laplacian_weight is None
        else (
            max(min(laplacian_weight[0], MAX_LAPLACIAN_WEIGHT), 0.0),
            max(min(laplacian_weight[1], MAX_LAPLACIAN_WEIGHT), 0.0),
        )
    )
    metric_weight = [
        [1.0 - laplacian_weight[0], laplacian_weight[0]],
        [1.0 - laplacian_weight[1], laplacian_weight[1]],
    ]

    wf = init_syn_sdc_wf(sloppy=sloppy, laplacian_weight=laplacian_weight)

    assert wf.inputs.syn.metric_weight == metric_weight


@pytest.mark.parametrize('sd_prior', [True, False])
def test_syn_preprocessing_wf_inputs(sd_prior):
    """Test appropriate instantiation of the SDC-SyN preprocessing workflow."""

    prep_wf = init_syn_preprocessing_wf(
        omp_nthreads=4,
        sd_prior=sd_prior,
        auto_bold_nss=True,
        t1w_inversion=True,
    )

    if not sd_prior:
        with pytest.raises(AttributeError):
            prep_wf.inputs.prior_msk.in_file
    else:
        assert prep_wf.inputs.prior_msk.thresh_low


@pytest.mark.parametrize('ants_version', ['2.2.0', '2.1.0', None])
def test_syn_wf_version(monkeypatch, ants_version):
    """Ensure errors are triggered with ANTs < 2.2."""
    from niworkflows.interfaces.fixes import FixHeaderRegistration as Registration

    monkeypatch.setattr(Registration, 'version', ants_version)
    if ants_version == '2.1.0':
        with pytest.raises(RuntimeError):
            init_syn_sdc_wf(debug=True, sloppy=True, omp_nthreads=4)
    else:
        wf = init_syn_sdc_wf(debug=True, sloppy=True, omp_nthreads=4)
        assert (ants_version or 'version unknown') in wf.__desc__


@pytest.mark.parametrize(
    'anat_res,epi_res,retval',
    [
        ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.8, 1.8, 1.8)),
        ((1.8, 1.8, 1.8), (2.0, 2.0, 2.0), (1.9, 1.9, 1.9)),
        ((1.5, 1.5, 1.5), (1.8, 1.8, 1.8), (1.8, 1.8, 1.8)),
        ((1.8, 1.8, 1.8), (2.5, 2.5, 2.5), (2.15, 2.15, 2.15)),
    ],
)
def test_adjust_zooms(anat_res, epi_res, retval, tmpdir, datadir):
    """Exercise the adjust zooms function node."""
    import nibabel as nb
    import numpy as np

    tmpdir.chdir()
    nb.Nifti1Image(
        np.zeros((10, 10, 10)),
        np.diag(list(anat_res) + [1]),
        None,
    ).to_filename('anat.nii.gz')
    nb.Nifti1Image(
        np.zeros((10, 10, 10)),
        np.diag(list(epi_res) + [1]),
        None,
    ).to_filename('epi.nii.gz')

    assert _adjust_zooms('anat.nii.gz', 'epi.nii.gz') == retval


@pytest.mark.parametrize(
    'in_dtype,out_dtype',
    [
        ('float32', 'int16'),
        ('int16', 'int16'),
        ('uint8', 'int16'),
        ('float64', 'int16'),
    ],
)
def test_ensure_dtype(in_dtype, out_dtype, tmpdir):
    """Exercise the set dtype function node."""
    import nibabel as nb
    import numpy as np

    tmpdir.chdir()
    nb.Nifti1Image(
        np.zeros((10, 10, 10), dtype=in_dtype),
        np.eye(4),
        None,
    ).to_filename(f'{in_dtype}.nii.gz')

    out_file = _set_dtype(f'{in_dtype}.nii.gz')
    if in_dtype == out_dtype:
        assert out_file == f'{in_dtype}.nii.gz'
    else:
        assert out_file == f'{in_dtype}_{out_dtype}.nii.gz'


def axcodes2aff(axcodes):
    """Return an affine matrix from axis codes."""
    return nb.orientations.inv_ornt_aff(
        nb.orientations.ornt_transform(
            nb.orientations.axcodes2ornt('RAS'),
            nb.orientations.axcodes2ornt(axcodes),
        ),
        (10, 10, 10),
    )


@pytest.mark.parametrize(
    ('fixed_ornt', 'moving_ornt', 'ijk', 'index'),
    [
        ('RAS', 'RAS', 'i', 0),
        ('RAS', 'RAS', 'j', 1),
        ('RAS', 'RAS', 'k', 2),
        ('RAS', 'PSL', 'i', 1),
        ('RAS', 'PSL', 'j', 2),
        ('RAS', 'PSL', 'k', 0),
        ('PSL', 'RAS', 'i', 2),
        ('PSL', 'RAS', 'j', 0),
        ('PSL', 'RAS', 'k', 1),
    ],
)
def test_mm2vox(tmp_path, fixed_ornt, moving_ornt, ijk, index):
    fixed_path = tmp_path / 'fixed.nii.gz'
    moving_path = tmp_path / 'moving.nii.gz'

    # Use separate zooms to make identifying the conversion easier
    fixed_aff = np.diag((2, 3, 4, 1))
    nb.save(
        nb.Nifti1Image(np.zeros((10, 10, 10)), axcodes2aff(fixed_ornt) @ fixed_aff),
        fixed_path,
    )
    nb.save(
        nb.Nifti1Image(np.zeros((10, 10, 10)), axcodes2aff(moving_ornt)),
        moving_path,
    )

    config = json.loads(data.load.readable('sd_syn.json').read_text())

    params = config['transform_parameters']
    mm_values = np.array([level[2] for level in params])

    vox_params = _mm2vox(str(moving_path), str(fixed_path), ijk, config)
    vox_values = [level[2] for level in vox_params]
    assert [mm_level[:2] == vox_level[:2] for mm_level, vox_level in zip(params, vox_params)]
    assert np.array_equal(vox_values, mm_values / [2, 3, 4][index])


@pytest.mark.parametrize(
    ('fixed_ornt', 'moving_ornt', 'ijk', 'index'),
    [
        ('RAS', 'RAS', 'i', 0),
        ('RAS', 'RAS', 'j', 1),
        ('RAS', 'RAS', 'k', 2),
        ('RAS', 'PSL', 'i', 1),
        ('RAS', 'PSL', 'j', 2),
        ('RAS', 'PSL', 'k', 0),
        ('PSL', 'RAS', 'i', 2),
        ('PSL', 'RAS', 'j', 0),
        ('PSL', 'RAS', 'k', 1),
    ],
)
def test_warp_dir(tmp_path, fixed_ornt, moving_ornt, ijk, index):
    fixed_path = tmp_path / 'fixed.nii.gz'
    moving_path = tmp_path / 'moving.nii.gz'

    nb.save(
        nb.Nifti1Image(np.zeros((10, 10, 10)), axcodes2aff(fixed_ornt)),
        fixed_path,
    )
    nb.save(
        nb.Nifti1Image(np.zeros((10, 10, 10)), axcodes2aff(moving_ornt)),
        moving_path,
    )

    for nlevels in range(1, 3):
        deformations = _warp_dir(str(moving_path), str(fixed_path), ijk, nlevels)
        assert len(deformations) == nlevels
        for val in deformations:
            assert val == [1.0 if i == index else 0.1 for i in range(3)]
