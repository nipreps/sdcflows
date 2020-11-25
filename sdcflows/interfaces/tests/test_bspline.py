"""Test B-Spline interfaces."""
import os
import numpy as np
import nibabel as nb
import pytest

from ..bspline import bspline_grid, Coefficients2Warp, BSplineApprox


@pytest.mark.parametrize("testnum", range(100))
def test_bsplines(tmp_path, testnum):
    """Test idempotency of B-Splines interpolation + approximation."""
    targetshape = (10, 12, 9)

    # Generate an oblique affine matrix for the target - it will be a common case.
    targetaff = nb.affines.from_matvec(
        nb.eulerangles.euler2mat(x=0.9, y=0.001, z=0.001) @ np.diag((2, 3, 4)),
    )

    # Intendedly mis-centered (exercise we may not have volume-centered NIfTIs)
    targetaff[:3, 3] = nb.affines.apply_affine(
        targetaff, 0.5 * (np.array(targetshape) - 3)
    )

    # Generate some target grid
    targetnii = nb.Nifti1Image(
        np.ones(targetshape),
        targetaff,
        None
    )
    targetnii.to_filename(tmp_path / "target.nii.gz")

    # Generate random coefficients
    gridnii = bspline_grid(targetnii, control_zooms_mm=(4, 6, 8))
    coeff = (np.random.random(size=gridnii.shape) - 0.5) * 500
    coeffnii = nb.Nifti1Image(coeff.astype("float32"), gridnii.affine, gridnii.header)
    coeffnii.to_filename(tmp_path / "coeffs.nii.gz")

    os.chdir(tmp_path)
    # Check that we can interpolate the coefficients on a target
    test1 = Coefficients2Warp(
        in_target=str(tmp_path / "target.nii.gz"),
        in_coeff=str(tmp_path / "coeffs.nii.gz"),
        pe_dir="j-",
        ro_time=1.0,
    ).run()

    # Approximate the interpolated target
    test2 = BSplineApprox(
        in_data=test1.outputs.out_field,
        in_mask=str(tmp_path / "target.nii.gz"),
        bs_spacing=[(4, 6, 8)],
        recenter=False,
        ridge_alpha=1e-4,
    ).run()

    # Absolute error of the interpolated field is always below 2 Hz
    assert np.all(np.abs(nb.load(test2.outputs.out_error).get_fdata()) < 2)
