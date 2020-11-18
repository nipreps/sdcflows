"""Test phase-difference type of fieldmaps."""
import os
from pathlib import Path
from json import loads

import pytest
from niworkflows.interfaces.bids import DerivativesDataSink
from nipype.pipeline import engine as pe

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

    wf = Workflow(
        name=f"phdiff_{fmap_path[0].name.replace('.nii.gz', '').replace('-', '_')}"
    )
    phdiff_wf = init_fmap_wf(omp_nthreads=2)
    phdiff_wf.inputs.inputnode.magnitude = [
        str(f.absolute()).replace("diff", "1").replace("phase", "magnitude")
        for f in fmap_path
    ]
    phdiff_wf.inputs.inputnode.fieldmap = [
        (str(f.absolute()), loads(Path(str(f).replace(".nii.gz", ".json")).read_text()))
        for f in fmap_path
    ]

    if outdir:
        from ...interfaces.reportlets import FieldmapReportlet

        rep = pe.Node(FieldmapReportlet(reference_label="Magnitude"), "simple_report")
        rep.interface._always_run = True

        ds_report = pe.Node(
            DerivativesDataSink(
                base_directory=str(outdir),
                out_path_base="sdcflows",
                datatype="figures",
                suffix="fieldmap",
                desc="phasediff",
                dismiss_entities="fmap",
            ),
            name="ds_report",
        )
        ds_report.inputs.source_file = str(fmap_path[0])

        dsink_fmap = pe.Node(
            DerivativesDataSink(
                base_directory=str(outdir),
                dismiss_entities="fmap",
                desc="phasediff",
                suffix="fieldmap",
            ),
            name="dsink_fmap",
        )
        dsink_fmap.interface.out_path_base = "sdcflows"
        dsink_fmap.inputs.source_file = str(fmap_path[0])

        # fmt: off
        wf.connect([
            (phdiff_wf, rep, [("outputnode.fmap", "fieldmap"),
                              ("outputnode.fmap_ref", "reference"),
                              ("outputnode.fmap_mask", "mask")]),
            (rep, ds_report, [("out_report", "in_file")]),
            (phdiff_wf, dsink_fmap, [("outputnode.fmap", "in_file")]),
        ])
        # fmt: on
    else:
        wf.add_nodes([phdiff_wf])

    if workdir:
        wf.base_dir = str(workdir)

    wf.run()
