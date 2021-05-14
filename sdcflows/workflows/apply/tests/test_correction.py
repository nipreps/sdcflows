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
"""Test unwarp."""
from pathlib import Path
from nipype.pipeline import engine as pe

from ...fit.fieldmap import init_magnitude_wf
from ..correction import init_unwarp_wf
from ..registration import init_coeff2epi_wf


def test_unwarp_wf(tmpdir, datadir, workdir, outdir):
    """Test the unwarping workflow."""
    distorted = (
        datadir
        / "HCP101006"
        / "sub-101006"
        / "func"
        / "sub-101006_task-rest_dir-LR_sbref.nii.gz"
    )

    magnitude = (
        datadir / "HCP101006" / "sub-101006" / "fmap" / "sub-101006_magnitude1.nii.gz"
    )
    fmap_ref_wf = init_magnitude_wf(2, name="fmap_ref_wf")
    fmap_ref_wf.inputs.inputnode.magnitude = magnitude

    epi_ref_wf = init_magnitude_wf(2, name="epi_ref_wf")
    epi_ref_wf.inputs.inputnode.magnitude = distorted

    reg_wf = init_coeff2epi_wf(2, debug=True, write_coeff=True)
    reg_wf.inputs.inputnode.fmap_coeff = [Path(__file__).parent / "fieldcoeff.nii.gz"]

    unwarp_wf = init_unwarp_wf(omp_nthreads=2, debug=True)
    unwarp_wf.inputs.inputnode.metadata = {
        "EffectiveEchoSpacing": 0.00058,
        "PhaseEncodingDirection": "i",
    }

    workflow = pe.Workflow(name="test_unwarp_wf")
    # fmt: off
    workflow.connect([
        (epi_ref_wf, unwarp_wf, [("outputnode.fmap_ref", "inputnode.distorted")]),
        (epi_ref_wf, reg_wf, [
            ("outputnode.fmap_ref", "inputnode.target_ref"),
            ("outputnode.fmap_mask", "inputnode.target_mask"),
        ]),
        (fmap_ref_wf, reg_wf, [
            ("outputnode.fmap_ref", "inputnode.fmap_ref"),
            ("outputnode.fmap_mask", "inputnode.fmap_mask"),
        ]),
        (reg_wf, unwarp_wf, [("outputnode.fmap_coeff", "inputnode.fmap_coeff")]),
    ])
    # fmt:on

    if outdir:
        from niworkflows.interfaces.reportlets.registration import (
            SimpleBeforeAfterRPT as SimpleBeforeAfter,
        )
        from ...outputs import DerivativesDataSink
        from ....interfaces.reportlets import FieldmapReportlet

        report = pe.Node(
            SimpleBeforeAfter(
                before_label="Distorted",
                after_label="Corrected",
            ),
            name="report",
            mem_gb=0.1,
        )
        ds_report = pe.Node(
            DerivativesDataSink(
                base_directory=str(outdir),
                suffix="bold",
                desc="corrected",
                datatype="figures",
                dismiss_entities=("fmap",),
                source_file=distorted,
            ),
            name="ds_report",
            run_without_submitting=True,
        )

        rep = pe.Node(FieldmapReportlet(apply_mask=True), "simple_report")
        rep.interface._always_run = True

        ds_fmap_report = pe.Node(
            DerivativesDataSink(
                base_directory=str(outdir),
                datatype="figures",
                suffix="bold",
                desc="fieldmap",
                dismiss_entities=("fmap",),
                source_file=distorted,
            ),
            name="ds_fmap_report",
        )

        # fmt: off
        workflow.connect([
            (epi_ref_wf, report, [("outputnode.fmap_ref", "before")]),
            (unwarp_wf, report, [("outputnode.corrected", "after"),
                                 ("outputnode.corrected_mask", "wm_seg")]),
            (report, ds_report, [("out_report", "in_file")]),
            (epi_ref_wf, rep, [("outputnode.fmap_ref", "reference"),
                               ("outputnode.fmap_mask", "mask")]),
            (unwarp_wf, rep, [("outputnode.fieldmap", "fieldmap")]),
            (rep, ds_fmap_report, [("out_report", "in_file")]),
        ])
        # fmt: on

    if workdir:
        workflow.base_dir = str(workdir)
    workflow.run(plugin="Linear")
