"""Test pepolar type of fieldmaps."""
import os
from pathlib import Path
from json import loads
import pytest
from niworkflows.interfaces.images import IntraModalMerge
from nipype.pipeline import engine as pe

from ..pepolar import init_topup_wf


@pytest.mark.skipif(os.getenv("TRAVIS") == "true", reason="this is TravisCI")
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

    wf = pe.Workflow(
        name=f"topup_{epi_path[0].name.replace('.nii.gz', '').replace('-', '_')}"
    )

    merge = pe.MapNode(IntraModalMerge(hmc=False), name="merge", iterfield=["in_files"])
    merge.inputs.in_files = in_data

    topup_wf = init_topup_wf(omp_nthreads=2, debug=True)
    metadata = [
        loads(Path(str(f).replace(".nii.gz", ".json")).read_text()) for f in in_data
    ]
    topup_wf.inputs.inputnode.metadata = metadata

    # fmt: off
    wf.connect([
        (merge, topup_wf, [("out_avg", "inputnode.in_data")]),
    ])
    # fmt: on

    if outdir:
        from nipype.interfaces.afni import Automask
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(outdir),
            write_coeff=True,
            custom_entities={"est": "pepolar"},
            bids_fmap_id="pepolar_id",
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = in_data
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = metadata

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(outdir), fmap_type="pepolar",
        )
        fmap_reports_wf.inputs.inputnode.source_files = in_data

        pre_mask = pe.Node(Automask(dilate=1, outputtype="NIFTI_GZ"), name="pre_mask")
        merge_corrected = pe.Node(IntraModalMerge(hmc=False), name="merge_corrected")

        # fmt: off
        wf.connect([
            (topup_wf, merge_corrected, [("outputnode.fmap_ref", "in_files")]),
            (merge_corrected, pre_mask, [("out_avg", "in_file")]),
            (merge_corrected, fmap_reports_wf, [("out_avg", "inputnode.fmap_ref")]),
            (topup_wf, fmap_reports_wf, [("outputnode.fmap", "inputnode.fieldmap")]),
            (pre_mask, fmap_reports_wf, [("out_file", "inputnode.fmap_mask")]),
            (topup_wf, fmap_derivatives_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
            ]),
        ])
        # fmt: on

    if workdir:
        wf.base_dir = str(workdir)

    wf.run(plugin="Linear")
