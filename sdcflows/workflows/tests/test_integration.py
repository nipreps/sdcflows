# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
"""Test the base workflow."""
import os
from pathlib import Path
import json
import pytest
import numpy as np
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from sdcflows import fieldmaps as sfm
from sdcflows.interfaces.reportlets import FieldmapReportlet
from sdcflows.workflows.apply import correction as swac
from niworkflows.interfaces.reportlets.registration import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="this is GH Actions")
@pytest.mark.parametrize("pe0", ["LR"])
def test_pepolar_wf(tmpdir, workdir, outdir, datadir, pe0):
    """Build a ``FieldmapEstimation`` workflow and test estimation and correction."""

    tmpdir.chdir()

    if not outdir:
        outdir = Path.cwd()

    pe1 = f"{pe0[::-1]}"

    datadir = datadir / "hcph-pilot_fieldmaps"

    metadata = json.loads(
        (datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_dwi.json").read_text()
    )

    # Generate an estimator workflow with the estimator object
    estimator = sfm.FieldmapEstimation(
        sources=[
            datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe1}_epi.nii.gz",
            datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-3dvolreg_dwi.nii.gz",
        ]
    )
    step1 = estimator.get_workflow(omp_nthreads=6, debug=False, sloppy=True)

    # Set inputs to estimator
    step1.inputs.inputnode.metadata = [
        json.loads(
            (datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe1}_epi.json").read_text()
        ),
        metadata,
    ]
    step1.inputs.inputnode.in_data = [
        str((datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe1}_epi.nii.gz").absolute()),
        str(
            (
                datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-3dvolreg_dwi.nii.gz"
            ).absolute()
        ),
    ]

    # Show a reportlet
    rpt_fieldmap = pe.Node(
        FieldmapReportlet(out_report=str(outdir / "test-integration_fieldmap.svg")),
        name="rpt_fieldmap",
    )

    unwarp_input = (
        datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-mockmotion_dwi.nii.gz"
    ).absolute()
    unwarp_xfms = np.load(
        datadir / f"sub-pilot_ses-15_acq-b0_dir-{pe0}_desc-mockmotion_dwi.npy"
    ).tolist()

    # Generate a warped reference for reportlet
    warped_ref = pe.Node(
        niu.Function(function=_transform_and_average), name="warped_ref"
    )
    warped_ref.inputs.in_file = str(unwarp_input)
    warped_ref.inputs.in_xfm = unwarp_xfms

    # Create an unwarp workflow and connect
    step2 = swac.init_unwarp_wf(
        omp_nthreads=6
    )  # six async threads should be doable on Circle
    step2.inputs.inputnode.metadata = metadata
    step2.inputs.inputnode.distorted = str(unwarp_input)
    step2.inputs.inputnode.hmc_xforms = unwarp_xfms

    # Write reportlet
    rpt_correct = pe.Node(
        SimpleBeforeAfter(
            after_label="Corrected",
            before_label="Distorted",
            out_report=str(outdir / "test-integration_sdc.svg"),
            dismiss_affine=True,
        ),
        name="rpt_correct",
    )

    wf = pe.Workflow(name=f"hcph_pepolar_{pe0}")

    # Execute in temporary directory
    wf.base_dir = f"{tmpdir}"
    wf.connect(
        [
            (
                step1,
                step2,
                [
                    ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                ],
            ),
            (
                step1,
                rpt_fieldmap,
                [
                    ("outputnode.fmap", "fieldmap"),
                    ("outputnode.fmap_ref", "reference"),
                    ("outputnode.fmap_ref", "moving"),
                ],
            ),
            (warped_ref, rpt_correct, [("out", "before")]),
            (
                step2,
                rpt_correct,
                [
                    ("outputnode.corrected_ref", "after"),
                ],
            ),
        ]
    )
    wf.run()


def _transform_and_average(in_file, in_xfm):
    import numpy as np
    from pathlib import Path
    from nitransforms.linear import LinearTransformsMapping
    from nipype.utils.filemanip import fname_presuffix

    out = fname_presuffix(
        in_file, suffix="_reference", newpath=str(Path.cwd().absolute())
    )

    realigned = LinearTransformsMapping(np.array(in_xfm), reference=in_file).apply(
        in_file
    )

    data = np.asanyarray(realigned.dataobj).mean(-1)

    realigned.__class__(
        data,
        realigned.affine,
        realigned.header,
    ).to_filename(out)

    return out
