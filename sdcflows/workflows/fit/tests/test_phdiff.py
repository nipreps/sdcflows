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
"""Test phase-difference type of fieldmaps."""
import os
from pathlib import Path
from json import loads

import pytest

from ..fieldmap import init_fmap_wf, Workflow


@pytest.mark.skipif(os.getenv("TRAVIS") == "true", reason="this is TravisCI")
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="this is GH Actions")
@pytest.mark.parametrize(
    "fmap_file",
    [
        ("ds001600/sub-1/fmap/sub-1_acq-v4_phasediff.nii.gz",),
        (
            "ds001600/sub-1/fmap/sub-1_acq-v2_phase1.nii.gz",
            "ds001600/sub-1/fmap/sub-1_acq-v2_phase2.nii.gz",
        ),
        (
            "ds001771/derivatives/openneuro/sub-36/fmap/sub-36_acq-topup1_fieldmap.nii.gz",
        ),
        ("HCP101006/sub-101006/fmap/sub-101006_phasediff.nii.gz",),
    ],
)
def test_phdiff(tmpdir, datadir, workdir, outdir, fmap_file):
    """Test creation of the workflow."""
    tmpdir.chdir()

    fmap_path = [datadir / f for f in fmap_file]
    fieldmaps = [
        (str(f.absolute()), loads(Path(str(f).replace(".nii.gz", ".json")).read_text()))
        for f in fmap_path
    ]

    wf = Workflow(
        name=f"phdiff_{fmap_path[0].name.replace('.nii.gz', '').replace('-', '_')}"
    )
    mode = "mapped" if "fieldmap" in fmap_path[0].name else "phasediff"
    phdiff_wf = init_fmap_wf(
        omp_nthreads=2,
        debug=True,
        sloppy=True,
        mode=mode,
    )
    phdiff_wf.inputs.inputnode.fieldmap = fieldmaps
    phdiff_wf.inputs.inputnode.magnitude = [
        f.replace("diff", "1")
        .replace("phase", "magnitude")
        .replace("fieldmap", "magnitude")
        for f, _ in fieldmaps
    ]

    if outdir:
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        outdir = outdir / "unittests" / fmap_file[0].split("/")[0]
        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(outdir),
            write_coeff=True,
            bids_fmap_id="phasediff_id",
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = [f for f, _ in fieldmaps]
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = [f for _, f in fieldmaps]

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(outdir),
            fmap_type=mode if len(fieldmaps) == 1 else "phases",
        )
        fmap_reports_wf.inputs.inputnode.source_files = [f for f, _ in fieldmaps]

        # fmt: off
        wf.connect([
            (phdiff_wf, fmap_reports_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask")]),
            (phdiff_wf, fmap_derivatives_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
            ]),
        ])
        # fmt: on
    else:
        wf.add_nodes([phdiff_wf])

    if workdir:
        wf.base_dir = str(workdir)

    wf.run(plugin="Linear")
