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
"""Test fieldmap interfaces."""
import numpy as np
import nibabel as nb
import pytest

from ..fmap import CheckB0Units


@pytest.mark.parametrize("units", ("rad/s", "Hz"))
def test_units(tmpdir, units):
    """Check the conversion of units."""
    tmpdir.chdir()
    hz = np.ones((5, 5, 5), dtype="float32") * 100
    data = hz.copy()

    if units == "rad/s":
        data *= 2.0 * np.pi

    nb.Nifti1Image(data, np.eye(4), None).to_filename("data.nii.gz")
    out_data = nb.load(
        CheckB0Units(units=units, in_file="data.nii.gz").run().outputs.out_file
    ).get_fdata(dtype="float32")

    assert np.allclose(hz, out_data)
