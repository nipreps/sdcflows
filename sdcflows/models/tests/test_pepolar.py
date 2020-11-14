"""Test pepolar type of fieldmaps."""
from pathlib import Path
from json import loads
import pytest
from niworkflows.interfaces.bids import DerivativesDataSink
from niworkflows.interfaces.images import IntraModalMerge
from nipype.pipeline import engine as pe

from ..pepolar import Workflow, init_topup_wf


@pytest.mark.parametrize(
    "epi_path",
    [
        # (
        #     "ds001600/sub-1/fmap/sub-1_dir-AP_epi.nii.gz",
        #     "ds001600/sub-1/fmap/sub-1_dir-PA_epi.nii.gz",
        # ),
        (
            "testdata/sub-HCP101006/fmap/sub-HCP101006_dir-LR_epi.nii.gz",
            "testdata/sub-HCP101006/fmap/sub-HCP101006_dir-RL_epi.nii.gz",
        ),
    ],
)
def test_topup_wf(tmpdir, datadir, workdir, outdir, epi_path):
    """Test preparation workflow."""
    epi_path = [datadir / f for f in epi_path]
    in_data = [str(f.absolute()) for f in epi_path]

    wf = Workflow(
        name=f"topup_{epi_path[0].name.replace('.nii.gz', '').replace('-', '_')}"
    )

    merge = pe.MapNode(IntraModalMerge(hmc=False), name="merge", iterfield=["in_files"])
    merge.inputs.in_files = in_data

    topup_wf = init_topup_wf(omp_nthreads=2, debug=True)
    topup_wf.inputs.inputnode.metadata = [
        loads(Path(str(f).replace(".nii.gz", ".json")).read_text()) for f in in_data
    ]

    # fmt: off
    wf.connect([
        (merge, topup_wf, [("out_avg", "inputnode.in_data")]),
    ])
    # fmt: on

    if outdir:
        from nipype.interfaces.afni import Automask
        from ...interfaces.reportlets import FieldmapReportlet

        pre_mask = pe.Node(Automask(dilate=1, outputtype="NIFTI_GZ"),
                           name="pre_mask")
        merge_corrected = pe.Node(IntraModalMerge(hmc=False), name="merge_corrected")

        rep = pe.Node(
            FieldmapReportlet(reference_label="EPI Reference"), "simple_report"
        )
        rep.interface._always_run = True
        ds_report = pe.Node(
            DerivativesDataSink(
                base_directory=str(outdir),
                out_path_base="sdcflows",
                datatype="figures",
                suffix="fieldmap",
                desc="pepolar",
                dismiss_entities="fmap",
            ),
            name="ds_report",
        )
        ds_report.inputs.source_file = in_data[0]

        # fmt: off
        wf.connect([
            (topup_wf, pre_mask, [("outputnode.fmap_ref", "in_file")]),
            (topup_wf, merge_corrected, [("outputnode.fmap_ref", "in_files")]),
            (merge_corrected, rep, [("out_avg", "reference")]),
            (topup_wf, rep, [("outputnode.fmap", "fieldmap")]),
            (pre_mask, rep, [("out_file", "mask")]),
            (rep, ds_report, [("out_report", "in_file")]),
        ])
        # fmt: on

    if workdir:
        wf.base_dir = str(workdir)

    wf.run()
