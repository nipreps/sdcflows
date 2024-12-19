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
"""Unit tests of the transform object."""
from subprocess import check_call
from itertools import product
import pytest
import numpy as np
import nibabel as nb
from nitransforms.linear import LinearTransformsMapping
from skimage.morphology import ball
import scipy.ndimage as nd
from nireports.interfaces.reporting.base import SimpleBeforeAfterRPT as SimpleBeforeAfter

from sdcflows import transform as tf
from sdcflows.interfaces.bspline import bspline_grid


def generate_oracle(
    coeff_file,
    rotation=(None, None, None),
    zooms=(2.0, 2.2, 1.5),
    flip=(False, False, False),
):
    """Generate an in-silico phantom, and a corresponding (aligned) B-Spline field."""
    data = ball(20)
    data[19:22, ...] = 0
    data = np.pad(data + nd.binary_erosion(data, ball(3)), 8)

    zooms = [z if not f else -z for z, f in zip(zooms, flip)]
    affine = np.diag(zooms + [1])
    affine[:3, 3] = -affine[:3, :3] @ ((np.array(data.shape) - 1) * 0.5)

    if any(rotation):
        R = nb.affines.from_matvec(
            nb.eulerangles.euler2mat(
                x=rotation[0],
                y=rotation[1],
                z=rotation[2],
            )
        )
        affine = R @ affine

    phantom_nii = nb.Nifti1Image(
        data.astype(np.uint8),
        affine,
        None,
    )

    # Generate the grid with our tools, but fill data with cached file
    coeff_data = nb.load(coeff_file).get_fdata(dtype="float32")
    coeff_nii = bspline_grid(
        phantom_nii,
        np.array(nb.load(coeff_file).header.get_zooms()),
    )
    coeff_nii = nb.Nifti1Image(
        coeff_data,
        coeff_nii.affine,
        coeff_nii.header,
    )
    return phantom_nii, coeff_nii


@pytest.mark.parametrize("pe_dir", ["j", "j-", "i", "i-", "k", "k-"])
@pytest.mark.parametrize("rotation", [(None, None, None), (0.2, None, None)])
@pytest.mark.parametrize("flip", list(product(*[(False, True)] * 3)))
def test_displacements_field(tmpdir, testdata_dir, outdir, pe_dir, rotation, flip):
    """Check the generated displacements fields."""
    tmpdir.chdir()

    # Generate test oracle
    phantom_nii, coeff_nii = generate_oracle(
        testdata_dir / "field-coeff-tests.nii.gz",
        rotation=rotation,
    )

    b0 = tf.B0FieldTransform(coeffs=coeff_nii)
    b0.apply(
        phantom_nii,
        pe_dir=pe_dir,
        ro_time=0.2,
        output_dtype="float32",
    ).to_filename("warped-sdcflows.nii.gz")
    b0.to_displacements(
        ro_time=0.2,
        pe_dir=pe_dir,
    ).to_filename("itk-displacements.nii.gz")

    phantom_nii.to_filename("phantom.nii.gz")
    # Run antsApplyTransform
    exit_code = check_call(
        [
            "antsApplyTransforms -d 3 -r phantom.nii.gz -i phantom.nii.gz "
            "-o warped-ants.nii.gz -n BSpline -t itk-displacements.nii.gz"
        ],
        shell=True,
    )
    assert exit_code == 0

    ours = np.asanyarray(nb.load("warped-sdcflows.nii.gz").dataobj)
    theirs = np.asanyarray(nb.load("warped-ants.nii.gz").dataobj)
    assert np.all((np.sqrt(((ours - theirs) ** 2).sum()) / ours.size) < 1e-1)

    if outdir:
        orientation = "".join([ax[bool(f)] for ax, f in zip(("RL", "AP", "SI"), flip)])

        SimpleBeforeAfter(
            after_label="Theirs (ANTs)",
            before_label="Ours (SDCFlows)",
            after="warped-ants.nii.gz",
            before="warped-sdcflows.nii.gz",
            out_report=str(
                outdir / f"xfm_pe-{pe_dir}_flip-{orientation}_x-{rotation[0] or 0}"
                f"_y-{rotation[1] or 0}_z-{rotation[2] or 0}.svg"
            ),
        ).run()


@pytest.mark.parametrize(
    "pe0",
    [
        "LR",
    ],
)
# @pytest.mark.parametrize("hmc", (True, False))
@pytest.mark.parametrize("hmc", (False, ))
@pytest.mark.parametrize("fmap", (True, False))
def test_apply_transform(tmpdir, outdir, datadir, pe0, hmc, fmap):
    """Test the .apply() member of the B0Transform object."""
    datadir = datadir / "hcph-pilot_fieldmaps"
    tmpdir.chdir()

    if not hmc and not fmap:
        return

    # Get coefficients file (at least for a quick reference if fmap is False)
    coeffs = [
        nb.load(
            datadir
            / f"sub-pilot_ses-15_desc-topup+coeff+{pe0}+{pe0[::-1]}_fieldmap.nii.gz"
        )
    ]

    if fmap is False:
        data = np.zeros(coeffs[0].shape, dtype=coeffs[0].header.get_data_dtype())

        # Replace coefficients file with all-zeros to test HMC is working
        coeffs[0] = coeffs[0].__class__(data, coeffs[0].affine, coeffs[0].header)

    warp = tf.B0FieldTransform(coeffs=coeffs)

    hmc_xfms = (
        np.load(datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-mockmotion_dwi.npy")
        if hmc
        else None
    )

    in_file = (
        datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-mockmotion_dwi.nii.gz"
        if hmc
        else datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-3dvolreg_dwi.nii.gz"
    )

    corrected = warp.apply(
        in_file,
        ro_time=0.0502149,
        pe_dir="i-",
        xfms=hmc_xfms,
        num_threads=6,
    )
    corrected.to_filename(f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-sdcflows_dwi.nii.gz")

    corrected.__class__(
        np.asanyarray(corrected.dataobj).mean(-1),
        corrected.affine,
        corrected.header,
    ).to_filename(f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-sdcflows_dwiref.nii.gz")

    error_margin = 0.5
    if fmap is False:  # If no fieldmap, this is equivalent to only HMC
        realigned = LinearTransformsMapping(hmc_xfms, reference=in_file).apply(in_file)
        error = np.sqrt(((corrected.dataobj - realigned.dataobj) ** 2))

        if outdir:
            # Do not include the first volume in the average to enhance differences
            realigned_data = np.asanyarray(corrected.dataobj)[..., 1:].mean(-1)
            realigned_data[realigned_data < 0] = 0
            realigned.__class__(
                realigned_data,
                realigned.affine,
                realigned.header,
            ).to_filename(
                f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-nitransforms_dwiref.nii.gz"
            )

            SimpleBeforeAfter(
                after_label="Theirs (3dvolreg)",
                before_label="Ours (SDCFlows)",
                after=f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-nitransforms_dwiref.nii.gz",
                before=f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-sdcflows_dwiref.nii.gz",
                out_report=str(
                    outdir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-justhmc_dwi.svg"
                ),
            ).run()

            realigned.__class__(error, realigned.affine, realigned.header,).to_filename(
                outdir
                / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-justhmc+error_dwi.nii.gz"
            )
    else:
        realigned = nb.load(in_file)
        error = np.nan_to_num(
            np.sqrt(((corrected.dataobj - realigned.dataobj) ** 2)), nan=0
        )
        error_margin = 200  # test oracle is pretty bad here - needs revision.

        if outdir:
            # Do not include the first volume in the average to enhance differences
            realigned_data = np.asanyarray(corrected.dataobj)[..., 1:].mean(-1)
            realigned_data[realigned_data < 0] = 0
            realigned.__class__(
                realigned_data,
                realigned.affine,
                realigned.header,
            ).to_filename(
                f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-3dvolreg_dwiref.nii.gz"
            )

            SimpleBeforeAfter(
                after_label="Theirs (NiTransforms)",
                before_label="Ours (SDCFlows)",
                after=f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-3dvolreg_dwiref.nii.gz",
                before=f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-sdcflows_dwiref.nii.gz",
                out_report=str(
                    outdir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-"
                    f"{'just' if not hmc else 'hmc+'}fmap_dwi.svg"
                ),
            ).run()

            realigned.__class__(error, realigned.affine, realigned.header,).to_filename(
                outdir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-"
                f"{'' if not hmc else 'hmc+'}fmap+error_dwi.nii.gz"
            )

    # error is below 0.5 in 95% of the voxels clipping margins
    assert np.percentile(error[2:-2, 2:-2, 2:-2], 95) < error_margin

    if outdir and fmap and hmc:
        # Generate a conversion without hmc
        corrected_nohmc = warp.apply(
            in_file,
            ro_time=0.0502149,
            pe_dir="i-",
            xfms=None,
            num_threads=6,
        )
        corrected_nohmc.to_filename(
            f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-sdcflows+nohmc_dwi.nii.gz"
        )

        # Do not include the first volume in the average to enhance differences
        corrected_nohmc.__class__(
            np.asanyarray(corrected.dataobj)[..., 1:].mean(-1),
            corrected.affine,
            corrected.header,
        ).to_filename(
            f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-sdcflows+nohmc_dwiref.nii.gz"
        )

        SimpleBeforeAfter(
            after_label="W/o HMC",
            before_label="With HMC",
            after=f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-sdcflows+nohmc_dwiref.nii.gz",
            before=f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-sdcflows_dwiref.nii.gz",
            out_report=str(
                outdir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-hmcdiff_dwi.svg"
            ),
        ).run()

        error = np.sqrt(((corrected.dataobj - corrected_nohmc.dataobj) ** 2))
        realigned.__class__(error, realigned.affine, realigned.header,).to_filename(
            outdir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-hmdiff+error_dwi.nii.gz"
        )


@pytest.mark.parametrize("pe_dir", ["j", "j-", "i", "i-", "k", "k-"])
def test_conversions(tmpdir, testdata_dir, pe_dir):
    """Check inverse functions."""
    tmpdir.chdir()

    fmap_nii = nb.load(testdata_dir / "topup-field.nii.gz")
    new_nii = tf.disp_to_fmap(
        tf.fmap_to_disp(
            fmap_nii,
            ro_time=0.2,
            pe_dir=pe_dir,
        ),
        fmap_nii,
        ro_time=0.2,
        pe_dir=pe_dir,
    )

    assert np.allclose(
        fmap_nii.get_fdata(dtype="float32"),
        new_nii.get_fdata(dtype="float32"),
    )


def test_grid_bspline_weights():
    target_shape = (10, 10, 10)
    target_aff = [[0.5, 0, 0, -2.5], [0, 0.5, 0, -2.5], [0, 0, 0.5, -2.5], [0, 0, 0, 1]]
    ctrl_shape = (4, 4, 4)
    ctrl_aff = [[3, 0, 0, -6], [0, 3, 0, -6], [0, 0, 3, -6], [0, 0, 0, 1]]

    weights = tf.grid_bspline_weights(
        nb.Nifti1Image(np.zeros(target_shape), target_aff),
        nb.Nifti1Image(np.zeros(ctrl_shape), ctrl_aff),
    ).tocsr()
    assert weights.shape == (1000, 64)
    # Empirically determined numbers intended to indicate that something
    # significant has changed. If it turns out we've been doing this wrong,
    # these numbers will probably change.
    assert np.isclose(weights[0, 0], 0.00089725334)
    assert np.isclose(weights[-1, -1], 0.18919244)
    assert np.isclose(weights.sum(axis=0).max(), 129.3907)
    assert np.isclose(weights.sum(axis=0).min(), 0.0052327816)
