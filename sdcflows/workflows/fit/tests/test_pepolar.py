"""Test pepolar type of fieldmaps."""
import os
from pathlib import Path
from json import loads
import pytest
from nipype.pipeline import engine as pe

from ..pepolar import init_topup_wf


@pytest.mark.skipif(os.getenv("TRAVIS") == "true", reason="this is TravisCI")
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="this is GH Actions")
@pytest.mark.parametrize(
    "epi_file",
    [
        (
            "ds001771/sub-36/fmap/sub-36_acq-topup1_dir-01_epi.nii.gz",
            "ds001771/sub-36/fmap/sub-36_acq-topup1_dir-02_epi.nii.gz",
        ),
        (
            "ds001771/sub-36/fmap/sub-36_acq-topup2_dir-01_epi.nii.gz",
            "ds001771/sub-36/fmap/sub-36_acq-topup2_dir-02_epi.nii.gz",
        ),
        (
            "HCP101006/sub-101006/fmap/sub-101006_dir-LR_epi.nii.gz",
            "HCP101006/sub-101006/fmap/sub-101006_dir-RL_epi.nii.gz",
        ),
    ],
)
def test_topup_wf(tmpdir, datadir, workdir, outdir, epi_file):
    """Test preparation workflow."""
    epi_path = [datadir / f for f in epi_file]
    in_data = [str(f.absolute()) for f in epi_path]

    wf = pe.Workflow(
        name=f"topup_{epi_path[0].name.replace('.nii.gz', '').replace('-', '_')}"
    )

    topup_wf = init_topup_wf(omp_nthreads=2, debug=True)
    metadata = [
        loads(Path(str(f).replace(".nii.gz", ".json")).read_text()) for f in in_data
    ]
    topup_wf.inputs.inputnode.in_data = in_data
    topup_wf.inputs.inputnode.metadata = metadata

    if outdir:
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        outdir = outdir / "unittests" / epi_file[0].split("/")[0]
        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(outdir),
            write_coeff=True,
            bids_fmap_id="pepolar_id",
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = in_data
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = metadata

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(outdir),
            fmap_type="pepolar",
        )
        fmap_reports_wf.inputs.inputnode.source_files = in_data

        # fmt: off
        wf.connect([
            (topup_wf, fmap_reports_wf, [("outputnode.fmap", "inputnode.fieldmap"),
                                         ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                                         ("outputnode.fmap_mask", "inputnode.fmap_mask")]),
            (topup_wf, fmap_derivatives_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
            ]),
        ])
        # fmt: on
    else:
        wf.add_nodes([topup_wf])

    if workdir:
        wf.base_dir = str(workdir)

    wf.run(plugin="Linear")
