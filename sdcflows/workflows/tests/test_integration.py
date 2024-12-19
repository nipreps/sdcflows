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

from pathlib import Path
import json
import pytest
import numpy as np
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from sdcflows import fieldmaps as sfm
from sdcflows.interfaces.reportlets import FieldmapReportlet
from sdcflows.workflows.apply import correction as swac
from sdcflows.workflows.apply import registration as swar
from nireports.interfaces.reporting.base import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)


@pytest.mark.slow
@pytest.mark.parametrize("pe0", ["LR", "PA"])
@pytest.mark.parametrize("mode", ["pepolar", "phasediff"])
def test_integration_wf(tmpdir, workdir, outdir, datadir, pe0, mode):
    """Build a ``FieldmapEstimation`` workflow and test estimation and correction."""

    tmpdir.chdir()

    if not outdir:
        outdir = Path.cwd()

    session = "15" if pe0 == "LR" else "14"

    pe1 = pe0[::-1]

    datadir = datadir / "hcph-pilot_fieldmaps"

    wf = pe.Workflow(name=f"hcph_{mode}_{pe0}")

    # Execute in temporary directory
    wf.base_dir = str(workdir or tmpdir)

    # Prepare some necessary data and metadata
    metadata = json.loads(
        (datadir / f"sub-pilot_ses-{session}_acq-b0_dir-{pe0}_dwi.json").read_text()
    )
    unwarp_input = (
        datadir / f"sub-pilot_ses-{session}_acq-b0_dir-{pe0}_desc-mockmotion_dwi.nii.gz"
    ).absolute()
    unwarp_xfms = np.load(
        datadir / f"sub-pilot_ses-{session}_acq-b0_dir-{pe0}_desc-mockmotion_dwi.npy"
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

    if mode == "pepolar":
        # Generate an estimator workflow with the estimator object
        estimator = sfm.FieldmapEstimation(
            sources=[
                datadir / f"sub-pilot_ses-{session}_acq-b0_dir-{pe1}_epi.nii.gz",
                datadir / f"sub-pilot_ses-{session}_acq-b0_dir-{pe0}_desc-3dvolreg_dwi.nii.gz",
            ],
        )
        step1 = estimator.get_workflow(omp_nthreads=6, debug=False, sloppy=True)

        # Set inputs to estimator
        step1.inputs.inputnode.metadata = [
            json.loads(
                (datadir / f"sub-pilot_ses-{session}_acq-b0_dir-{pe1}_epi.json").read_text()
            ),
            metadata,
        ]
        step1.inputs.inputnode.in_data = [
            str((datadir / f"sub-pilot_ses-{session}_acq-b0_dir-{pe1}_epi.nii.gz").absolute()),
            str(
                (
                    datadir
                    / f"sub-pilot_ses-{session}_acq-b0_dir-{pe0}_desc-3dvolreg_dwi.nii.gz"
                ).absolute()
            ),
        ]
    else:
        # Generate an estimator workflow with the estimator object
        estimator = sfm.FieldmapEstimation(
            sources=[datadir / f"sub-pilot_ses-{session}_phasediff.nii.gz"],
        )
        step1 = estimator.get_workflow(omp_nthreads=6, debug=False)

        coeff2epi_wf = swar.init_coeff2epi_wf(omp_nthreads=4, sloppy=True, debug=True)
        coeff2epi_wf.inputs.inputnode.target_mask = str(
            (
                datadir
                / f"sub-pilot_ses-{session}_acq-b0_dir-{pe0}_desc-aftersdcbrain_mask.nii.gz"
            ).absolute()
        )

        # Check fmap2epi alignment
        rpt_coeff2epi = pe.Node(
            SimpleBeforeAfter(
                after_label="GRE (mag)",
                before_label="EPI (ref)",
                out_report=str(
                    outdir / f"sub-pilot_ses-{session}_desc-aligned+{pe0}_fieldmap.svg"
                ),
                dismiss_affine=True,
            ),
            name="rpt_coeff2epi",
        )

        # fmt:off
        wf.connect([
            (step1, coeff2epi_wf, [
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
            ]),
            (warped_ref, coeff2epi_wf, [("out", "inputnode.target_ref")]),
            (coeff2epi_wf, step2, [
                ("outputnode.target2fmap_xfm", "inputnode.data2fmap_xfm"),
            ]),
            (coeff2epi_wf, rpt_coeff2epi, [("coregister.warped_image", "before")]),
            (step1, rpt_coeff2epi, [("outputnode.fmap_ref", "after")]),
        ])
        # fmt:on

    # Show a reportlet
    rpt_fieldmap = pe.Node(
        FieldmapReportlet(
            out_report=str(
                outdir / f"sub-pilot_ses-{session}_desc-{mode}+{pe0}_fieldmap.svg"
            ),
        ),
        name="rpt_fieldmap",
    )

    # Write reportlet
    rpt_correct = pe.Node(
        SimpleBeforeAfter(
            after_label="Corrected",
            before_label="Distorted",
            out_report=str(outdir / f"sub-pilot_ses-{session}_desc-{mode}+{pe0}_dwi.svg"),
            # dismiss_affine=True,
        ),
        name="rpt_correct",
    )

    # fmt:off
    wf.connect([
        (step1, step2, [
            ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
        ]),
        (step1, rpt_fieldmap, [
            ("outputnode.fmap", "fieldmap"),
            ("outputnode.fmap_ref", "reference"),
            ("outputnode.fmap_ref", "moving"),
        ]),
        (warped_ref, rpt_correct, [("out", "before")]),
        (step2, rpt_correct, [
            ("outputnode.corrected_ref", "after"),
        ]),
    ])
    # fmt:on
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
