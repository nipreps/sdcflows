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
"""Test utilities."""
import pytest
import numpy as np
import nibabel as nb

from ..utils import Flatten, ConvertWarp, Deoblique, Reoblique, PadSlices


def test_Flatten(tmpdir):
    """Test the flattening interface."""
    tmpdir.chdir()
    shape = (5, 5, 5)
    nb.Nifti1Image(np.zeros(shape), np.eye(4), None).to_filename("file1.nii.gz")
    nb.Nifti1Image(np.zeros((*shape, 6)), np.eye(4), None).to_filename("file2.nii.gz")
    nb.Nifti1Image(np.zeros((*shape, 2)), np.eye(4), None).to_filename("file3.nii.gz")

    out = Flatten(
        in_data=["file1.nii.gz", "file2.nii.gz", "file3.nii.gz"],
        in_meta=[{"a": 1}, {"b": 2}, {"c": 3}],
        max_trs=3,
    ).run()

    assert len(out.outputs.out_list) == 6

    out_meta = out.outputs.out_meta
    assert out_meta[0] == {"a": 1}
    assert out_meta[1] == out_meta[2] == out_meta[3] == {"b": 2}
    assert out_meta[4] == out_meta[5] == {"c": 3}


@pytest.mark.parametrize("shape", [(10, 10, 10, 1, 3), (10, 10, 10, 3)])
def test_ConvertWarp(tmpdir, shape):
    """Exercise the interface."""
    tmpdir.chdir()

    nb.Nifti1Image(np.zeros(shape, dtype="uint8"), np.eye(4), None).to_filename(
        "3dQwarp.nii.gz"
    )

    out = ConvertWarp(in_file="3dQwarp.nii.gz").run()

    nii = nb.load(out.outputs.out_file)
    assert nii.header.get_data_dtype() == np.float32
    assert nii.header.get_intent() == ("vector", (), "")
    assert nii.shape == (10, 10, 10, 1, 3)


@pytest.mark.parametrize(
    "angles,oblique",
    [
        ((0, 0, 0), False),
        ((0.9, 0.001, 0.001), True),
        ((0, 0, 2 * np.pi), False),
    ],
)
def test_Xeoblique(tmpdir, angles, oblique):
    """Exercise De/Reoblique interfaces."""
    tmpdir.chdir()

    affine = nb.affines.from_matvec(nb.eulerangles.euler2mat(*angles))
    nb.Nifti1Image(np.zeros((10, 10, 10), dtype="uint8"), affine, None).to_filename(
        "epi.nii.gz"
    )

    result = (
        Deoblique(
            in_file="epi.nii.gz",
            in_mask="epi.nii.gz",
        )
        .run()
        .outputs
    )

    assert np.allclose(nb.load(result.out_file).affine, affine) is not oblique

    reoblique = (
        Reoblique(
            in_plumb=result.out_file,
            in_field=result.out_file,
            in_epi="epi.nii.gz",
        )
        .run()
        .outputs
    )

    assert np.allclose(nb.load(reoblique.out_epi).affine, affine)


@pytest.mark.parametrize("in_shape,expected_shape,padded", [
    ((2,2,2), (2,2,2), False),
    ((2,2,3), (2,2,4), True),
    ((3,3,2,2), (3,3,2,2), False),
    ((3,3,3,2), (3,3,4,2), True),
])
def test_pad_slices(tmpdir, in_shape, expected_shape, padded):
    tmpdir.chdir()

    data = np.random.rand(*in_shape)
    aff = np.eye(4)

    # RAS
    img = nb.Nifti1Image(data, aff)
    img.to_filename("epi-ras.nii.gz")
    res = PadSlices(in_file="epi-ras.nii.gz").run().outputs

    # LPS
    newaff = aff.copy()
    newaff[0, 0] *= -1.0
    newaff[1, 1] *= -1.0
    newaff[:2, 3] = aff.dot(np.hstack((np.array(img.shape[:3]) - 1, 1.0)))[:2]
    img2 = nb.Nifti1Image(np.flip(np.flip(data, 0), 1), newaff)
    img2.to_filename("epi-lps.nii.gz")
    res2 = PadSlices(in_file="epi-lps.nii.gz").run().outputs


    out_ras = nb.load(res.out_file)
    out_lps = nb.load(res2.out_file)
    assert out_ras.shape == out_lps.shape == expected_shape
    assert res.padded == padded
