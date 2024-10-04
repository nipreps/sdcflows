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
"""Test B-Spline interfaces."""
import os
import numpy as np
import nibabel as nb
import pytest

from ..bspline import (
    bspline_grid,
    ApplyCoeffsField,
    BSplineApprox,
    TOPUPCoeffReorient,
    _fix_topup_fieldcoeff,
)

rng = np.random.default_rng(seed=20160305)  # First commit in nipreps/sdcflows


@pytest.mark.parametrize("testnum", range(100))
def test_bsplines(tmp_path, testnum):
    """Test idempotency of B-Splines interpolation + approximation."""
    targetshape = (50, 50, 30)

    # Generate an oblique affine matrix for the target - it will be a common case.
    targetaff = nb.affines.from_matvec(
        nb.eulerangles.euler2mat(x=-0.9, y=0.001, z=0.001) @ np.diag((2, 2, 2.4)),
    )

    # Intendedly mis-centered (exercise we may not have volume-centered NIfTIs)
    targetaff[:3, 3] = nb.affines.apply_affine(
        targetaff, 0.5 * (np.array(targetshape) - 3)
    )

    mask = np.zeros(targetshape)
    mask[10:-10, 10:-10, 6:-6] = 1
    # Generate some target grid
    targetnii = nb.Nifti1Image(mask, targetaff, None)
    targetnii.header.set_qform(targetaff, code=1)
    targetnii.header.set_sform(targetaff, code=1)
    targetnii.to_filename(tmp_path / "mask.nii.gz")

    # Generate random coefficients
    gridnii = bspline_grid(targetnii, control_zooms_mm=(40, 40, 16))
    coeff = (rng.standard_normal(size=gridnii.shape)) * 100
    coeffnii = nb.Nifti1Image(coeff.astype("float32"), gridnii.affine, gridnii.header)
    coeffnii.header["cal_max"] = np.abs(coeff).max()
    coeffnii.header["cal_min"] = -coeffnii.header["cal_max"]
    coeffnii.header.set_qform(gridnii.affine, code=1)
    coeffnii.header.set_sform(gridnii.affine, code=1)
    coeffnii.to_filename(tmp_path / "coeffs.nii.gz")

    os.chdir(tmp_path)
    # Check that we can interpolate the coefficients on a target
    test1 = ApplyCoeffsField(
        in_data=str(tmp_path / "mask.nii.gz"),
        in_coeff=str(tmp_path / "coeffs.nii.gz"),
        pe_dir="j-",
        ro_time=1.0,
    ).run()

    fieldnii = nb.load(test1.outputs.out_field)
    fielddata = fieldnii.get_fdata()
    fielddata -= np.median(fielddata)
    fielddata = 200 * fielddata / np.abs(fielddata).max()

    fieldnii.header["cal_max"] = np.abs(fielddata).max()
    fieldnii.header["cal_min"] = -fieldnii.header["cal_max"]
    fieldnii.header.set_qform(targetaff, code=1)
    fieldnii.header.set_sform(targetaff, code=1)

    nb.Nifti1Image(fielddata, targetaff, fieldnii.header).to_filename(
        tmp_path / "testfield.nii.gz",
    )

    # Approximate the interpolated target
    test2 = BSplineApprox(
        in_data=str(tmp_path / "testfield.nii.gz"),
        # in_mask=str(tmp_path / "mask.nii.gz"),
        bs_spacing=[(40, 40, 16)],
        zooms_min=0,
        recenter=False,
        ridge_alpha=1e-4,
    ).run()

    # Absolute error of the interpolated field
    # TODO - this is probably too high. We need to revisit these tests.
    error = nb.load(test2.outputs.out_error).get_fdata()
    assert (np.abs(error) > 25).sum() / error.size < 0.05  # 95% of errors below 25 Hz


def test_topup_coeffs(tmpdir, testdata_dir):
    """Check the revision of TOPUP headers."""
    tmpdir.chdir()
    result = TOPUPCoeffReorient(
        in_coeff=str(testdata_dir / "topup-coeff.nii.gz"),
        fmap_ref=str(testdata_dir / "epi.nii.gz"),
        pe_dir="j",
    ).run()

    nii = nb.load(result.outputs.out_coeff)
    ctrl = nb.load(testdata_dir / "topup-coeff-fixed.nii.gz")
    assert np.allclose(nii.affine, ctrl.affine)

    nb.Nifti1Image(nii.get_fdata()[:-1, :-1, :-1], nii.affine, nii.header).to_filename(
        "failing.nii.gz"
    )

    with pytest.raises(ValueError):
        TOPUPCoeffReorient(
            in_coeff="failing.nii.gz",
            fmap_ref=str(testdata_dir / "epi.nii.gz"),
            pe_dir="j",
        ).run()

    # Test automatic output file name generation, just for coverage
    with pytest.raises(ValueError):
        _fix_topup_fieldcoeff("failing.nii.gz", str(testdata_dir / "epi.nii.gz"), "i")


def test_topup_coeffs_interpolation(tmpdir, testdata_dir):
    """Check that our interpolation is not far away from TOPUP's."""
    tmpdir.chdir()
    result = ApplyCoeffsField(
        in_data=str(testdata_dir / "epi.nii.gz"),
        in_coeff=str(testdata_dir / "topup-coeff-fixed.nii.gz"),
        pe_dir="j-",
        ro_time=1.0,
    ).run()
    interpolated = nb.as_closest_canonical(
        nb.load(result.outputs.out_field)
    ).get_fdata()
    reference = nb.as_closest_canonical(
        nb.load(testdata_dir / "topup-field.nii.gz")
    ).get_fdata()
    assert np.sqrt(np.mean((interpolated - reference) ** 2)) < 3
