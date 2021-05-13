"""Test pepolar type of fieldmaps."""
import os
from pathlib import Path
from json import loads
import pytest
from nipype.pipeline import engine as pe

from ..pepolar import init_topup_wf


@pytest.mark.skipif(os.getenv("TRAVIS") == "true", reason="this is TravisCI")
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="this is GH Actions")
@pytest.mark.parametrize("ds", ("ds001771", "HCP101006"))
def test_topup_wf(tmpdir, bids_layouts, workdir, outdir, ds):
    """Test preparation workflow."""
    layout = bids_layouts[ds]
    epi_path = sorted(
        layout.get(suffix="epi", extension=["nii", "nii.gz"], scope="raw"),
        key=lambda k: k.path,
    )
    in_data = [f.path for f in epi_path]

    wf = pe.Workflow(name=f"topup_{ds}")
    topup_wf = init_topup_wf(omp_nthreads=2, debug=True, sloppy=True)
    metadata = [layout.get_metadata(f.path) for f in epi_path]

    topup_wf.inputs.inputnode.in_data = in_data
    topup_wf.inputs.inputnode.metadata = metadata

    if outdir:
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        outdir = outdir / "unittests" / f"topup_{ds}"
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
