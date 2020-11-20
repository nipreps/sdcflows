"""Test pepolar type of fieldmaps."""
import os
import pytest
from nipype.pipeline import engine as pe

from ...fit.fieldmap import init_magnitude_wf
from ..registration import init_coeff2epi_wf


@pytest.mark.skipif(os.getenv("TRAVIS") == "true", reason="this is TravisCI")
def test_registration_wf(tmpdir, datadir, workdir, outdir):
    """Test fieldmap-to-target alignment workflow."""
    epi_ref_wf = init_magnitude_wf(2, name="epi_ref_wf")
    epi_ref_wf.inputs.inputnode.magnitude = (
        datadir
        / "testdata"
        / "sub-HCP101006"
        / "func"
        / "sub-HCP101006_task-rest_dir-LR_sbref.nii.gz"
    )

    magnitude = (
        datadir
        / "testdata"
        / "sub-HCP101006"
        / "fmap"
        / "sub-HCP101006_magnitude1.nii.gz"
    )
    fmap_ref_wf = init_magnitude_wf(2, name="fmap_ref_wf")
    fmap_ref_wf.inputs.inputnode.magnitude = magnitude

    reg_wf = init_coeff2epi_wf(2, debug=True)

    workflow = pe.Workflow(name="test_registration_wf")
    workflow.connect(
        [
            (
                epi_ref_wf,
                reg_wf,
                [
                    ("outputnode.fmap_ref", "inputnode.target_ref"),
                    ("outputnode.fmap_mask", "inputnode.target_mask"),
                ],
            ),
            (
                fmap_ref_wf,
                reg_wf,
                [
                    ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                    ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                ],
            ),
        ]
    )

    if outdir:
        from niworkflows.interfaces import SimpleBeforeAfter
        from ...outputs import DerivativesDataSink

        report = pe.Node(
            SimpleBeforeAfter(before_label="Target EPI", after_label="B0 Reference",),
            name="report",
            mem_gb=0.1,
        )
        ds_report = pe.Node(
            DerivativesDataSink(
                base_directory=str(outdir),
                suffix="fieldmap",
                space="sbref",
                datatype="figures",
                dismiss_entities=("fmap",),
                source_file=magnitude,
            ),
            name="ds_report",
            run_without_submitting=True,
        )

        workflow.connect(
            [
                (epi_ref_wf, report, [("outputnode.fmap_ref", "before")]),
                (reg_wf, report, [("outputnode.fmap_ref", "after")]),
                (report, ds_report, [("out_report", "in_file")]),
            ]
        )

    if workdir:
        workflow.base_dir = str(workdir)

    workflow.run(plugin="Linear")
