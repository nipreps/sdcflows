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
"""Test the fieldmap reportlets."""
from pathlib import Path
import pytest
from ..reportlets import FieldmapReportlet
from ...utils.epimanip import epi_mask


@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("apply_mask", [True, False])
def test_FieldmapReportlet(tmpdir, outdir, testdata_dir, mask, apply_mask):
    """Generate one reportlet."""
    tmpdir.chdir()

    if not outdir:
        outdir = Path.cwd()

    report = FieldmapReportlet(
        reference=str(testdata_dir / "epi.nii.gz"),
        moving=str(testdata_dir / "epi.nii.gz"),
        fieldmap=str(testdata_dir / "topup-field.nii.gz"),
        out_report=str(
            outdir / f"test-fieldmap{'-masked' * mask}{'-zeroed' * apply_mask}.svg"
        ),
        apply_mask=apply_mask,
    )

    if mask:
        report.inputs.mask = epi_mask(str(testdata_dir / "epi.nii.gz"))

    report.run()
