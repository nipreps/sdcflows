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
@pytest.mark.parametrize("factor", [1, 4, 0.5])
def test_deoblique_and_zooms(padding, factor):
    """Check deoblique and denser."""

    # Generate an example reference image
    ref_data = np.zeros((20, 30, 40), dtype=np.float32)
    ref_affine = np.diag([1.0, 1.2, 0.8, 1.0])
    ref_img = nb.Nifti1Image(ref_data, ref_affine)
    ref_zooms = np.array(ref_img.header.get_zooms()[:3])

    # Generate an example oblique image
    ob_data = np.ones_like(ref_data)

    # Rotate 90 degrees around x-axis
    rotate = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    ob_img = nb.Nifti1Image(ob_data, rotate @ ref_affine)

    # Call function with default parameters
    out_img = deoblique_and_zooms(ref_img, ob_img, padding=padding, factor=factor)

    # Check output shape and zooms
    assert np.allclose(out_img.header.get_zooms()[:3], ref_zooms / factor)

    resampled = Affine(reference=out_img).apply(ob_img, order=0)
    ref_volume = ref_data.size * ref_zooms.prod()
    res_volume = resampled.get_fdata().sum() * np.prod(resampled.header.get_zooms())
    # 20% of change in volume is too high, must be an error somewhere
    assert abs(ref_volume - res_volume) < ref_volume * 0.2
