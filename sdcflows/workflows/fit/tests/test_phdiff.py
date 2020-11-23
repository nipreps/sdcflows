"""Test phase-difference type of fieldmaps."""
import os
from pathlib import Path
from json import loads

import pytest

from ..fieldmap import init_fmap_wf, Workflow


@pytest.mark.skipif(os.getenv("TRAVIS") == "true", reason="this is TravisCI")
@pytest.mark.parametrize(
    "fmap_path",
    [
        ("ds001600/sub-1/fmap/sub-1_acq-v4_phasediff.nii.gz",),
        (
            "ds001600/sub-1/fmap/sub-1_acq-v2_phase1.nii.gz",
            "ds001600/sub-1/fmap/sub-1_acq-v2_phase2.nii.gz",
        ),
        ("testdata/sub-HCP101006/fmap/sub-HCP101006_phasediff.nii.gz",),
    ],
)
def test_phdiff(tmpdir, datadir, workdir, outdir, fmap_path):
    """Test creation of the workflow."""
    tmpdir.chdir()

    fmap_path = [datadir / f for f in fmap_path]
    fieldmaps = [
        (str(f.absolute()), loads(Path(str(f).replace(".nii.gz", ".json")).read_text()))
        for f in fmap_path
    ]

    wf = Workflow(
        name=f"phdiff_{fmap_path[0].name.replace('.nii.gz', '').replace('-', '_')}"
    )
    phdiff_wf = init_fmap_wf(omp_nthreads=2, debug=True)
    phdiff_wf.inputs.inputnode.fieldmap = fieldmaps
    phdiff_wf.inputs.inputnode.magnitude = [
        f.replace("diff", "1").replace("phase", "magnitude") for f, _ in fieldmaps
    ]

    if outdir:
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(outdir),
            write_coeff=True,
            custom_entities={"est": "phasediff"},
            bids_fmap_id="phasediff_id",
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = [f for f, _ in fieldmaps]
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = [f for _, f in fieldmaps]

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(outdir),
            fmap_type="phasediff" if len(fieldmaps) == 1 else "phases",
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
