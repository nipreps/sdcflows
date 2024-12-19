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
import json
import pytest
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nireports.interfaces.reporting.base import SimpleBeforeAfterRPT as SimpleBeforeAfter
from sdcflows.workflows.apply.correction import init_unwarp_wf


@pytest.mark.parametrize("with_affine", [False, True])
def test_unwarp_wf(tmpdir, datadir, workdir, outdir, with_affine):
    """Test the unwarping workflow."""
    tmpdir.chdir()

    derivs_path = datadir / "HCP101006" / "derivatives" / "sdcflows-2.x"

    distorted = (
        datadir
        / "HCP101006"
        / "sub-101006"
        / "func"
        / "sub-101006_task-rest_dir-LR_sbref.nii.gz"
    )

    workflow = init_unwarp_wf(omp_nthreads=2, debug=True)
    workflow.inputs.inputnode.distorted = str(distorted)
    workflow.inputs.inputnode.metadata = json.loads(
        (distorted.parent / distorted.name.replace(".nii.gz", ".json")).read_text()
    )
    workflow.inputs.inputnode.fmap_coeff = [
        str(derivs_path / "sub-101006_coeff-1_desc-topup_fieldmap.nii.gz")
    ]

    if with_affine:
        workflow.inputs.inputnode.fmap2data_xfm = str(
            str(derivs_path / "sub-101006_from-sbrefLR_to-fieldmapref_mode-image_xfm.mat")
        )

    if outdir:
        from ...outputs import DerivativesDataSink
        from ....interfaces.reportlets import FieldmapReportlet

        outdir = outdir / f"with{'' if with_affine else 'out'}-affine"
        outdir.mkdir(exist_ok=True, parents=True)
        unwarp_wf = workflow  # Change variable name
        workflow = pe.Workflow(name="outputs_unwarp_wf")
        squeeze = pe.Node(niu.Function(function=_squeeze), name="squeeze")

        report = pe.Node(
            SimpleBeforeAfter(
                before_label="Distorted",
                after_label="Corrected",
                before=str(distorted)
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

        rep = pe.Node(FieldmapReportlet(), "simple_report")
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
            (unwarp_wf, squeeze, [("outputnode.corrected", "in_file")]),
            (unwarp_wf, report, [("outputnode.corrected_mask", "wm_seg")]),
            (squeeze, report, [("out", "after")]),
            (report, ds_report, [("out_report", "in_file")]),
            (squeeze, rep, [("out", "reference")]),
            (unwarp_wf, rep, [
                ("outputnode.fieldmap", "fieldmap"),
                ("outputnode.corrected_mask", "mask"),
            ]),
            (rep, ds_fmap_report, [("out_report", "in_file")]),
        ])
        # fmt: on

    if workdir:
        workflow.base_dir = str(workdir)
    workflow.run(plugin="Linear")


def _squeeze(in_file):
    from pathlib import Path
    import nibabel as nb

    img = nb.load(in_file)
    squeezed = nb.squeeze_image(img)

    if squeezed.shape == img.shape:
        return in_file

    out_fname = Path.cwd() / Path(in_file).name
    squeezed.to_filename(out_fname)
    return str(out_fname)
