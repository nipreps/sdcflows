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
"""Test EPI manipulation routines."""
import numpy as np
import pytest
import nibabel as nb
from nitransforms.linear import Affine
from sdcflows.utils.tools import brain_masker, deoblique_and_zooms


def test_epi_mask(tmpdir, testdata_dir):
    """Check mask algorithm."""
    tmpdir.chdir()
    mask = brain_masker(testdata_dir / "epi.nii.gz")[-1]
    assert abs(np.asanyarray(nb.load(mask).dataobj).sum() - 166476) < 10


@pytest.mark.parametrize("padding", [0, 1, 4])
@pytest.mark.parametrize("factor", [1, 4, 0.8])
@pytest.mark.parametrize("centered", [True, False])
@pytest.mark.parametrize(
    "rotate",
    [
        np.eye(4),
        # Rotate 90 degrees around x-axis
        np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
        # Rotate 30 degrees around x-axis
        nb.affines.from_matvec(
            nb.eulerangles.euler2mat(0, 0, 30 * np.pi / 180),
            (0, 0, 0),
        ),
        # Rotate 48 degrees around y-axis and translation
        nb.affines.from_matvec(
            nb.eulerangles.euler2mat(0, 48 * np.pi / 180, 0),
            (2.0, 1.2, 0.7),
        ),
    ],
)
def test_deoblique_and_zooms(tmpdir, padding, factor, centered, rotate, debug=False):
    """Check deoblique and denser."""
    tmpdir.chdir()

    # Generate an example reference image
    ref_shape = (80, 128, 160) if factor < 1 else (20, 32, 40)
    ref_data = np.zeros(ref_shape, dtype=np.float32)
    ref_data[1:-2, 10:-11, 1:-2] = 1
    ref_affine = np.diag([2.0, 1.25, 1.0, 1.0])
    ref_zooms = nb.affines.voxel_sizes(ref_affine)
    if centered:
        ref_affine[:3, 3] -= nb.affines.apply_affine(
            ref_affine,
            0.5 * (np.array(ref_data.shape) - 1),
        )
    ref_img = nb.Nifti1Image(ref_data, ref_affine)
    ref_img.header.set_qform(ref_affine, 1)
    ref_img.header.set_sform(ref_affine, 0)

    # Generate an example oblique image
    mov_affine = rotate @ ref_affine
    mov_img = nb.Nifti1Image(ref_data, mov_affine)
    mov_img.header.set_qform(mov_affine, 1)
    mov_img.header.set_sform(mov_affine, 0)

    # Call function with default parameters
    out_img = deoblique_and_zooms(ref_img, mov_img, padding=padding, factor=factor)

    # Check output shape and zooms
    assert np.allclose(out_img.header.get_zooms()[:3], ref_zooms / factor)

    # Check idempotency with a lot of tolerance
    ref_resampled = Affine(reference=out_img).apply(ref_img, order=0)
    ref_back = Affine(reference=ref_img).apply(ref_resampled, order=0)
    resampled = Affine(reference=out_img).apply(mov_img, order=2)
    if debug:
        ref_img.to_filename("reference.nii.gz")
        ref_back.to_filename("reference_back.nii.gz")
        ref_resampled.to_filename("reference_resampled.nii.gz")
        mov_img.to_filename("moving.nii.gz")
        resampled.to_filename("resampled.nii.gz")

    # Allow up to 3% pixels wrong after up(down)sampling and walk back.
    assert (
        np.abs(np.clip(ref_back.get_fdata(), 0, 1) - ref_data).sum()
        < ref_data.size * 0.03
    )

    vox_vol_out = np.prod(out_img.header.get_zooms())
    vox_vol_mov = np.prod(mov_img.header.get_zooms())
    vol_factor = vox_vol_out / vox_vol_mov

    ref_volume = ref_data.sum()
    res_volume = np.clip(resampled.get_fdata(), 0, 1).sum() * vol_factor
    # Tolerate up to 2% variation of the volume of the moving image
    assert abs(1 - res_volume / ref_volume) < 0.02
