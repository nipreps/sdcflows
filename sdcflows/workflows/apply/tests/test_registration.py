"""Test pepolar type of fieldmaps."""
import os
import pytest

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from ...fit.fieldmap import init_magnitude_wf
from ..registration import init_coeff2epi_wf


@pytest.mark.skipif(os.getenv("TRAVIS") == "true", reason="this is TravisCI")
def test_registration_wf(tmpdir, datadir, workdir, outdir):
    """Test fieldmap-to-target alignment workflow."""
    epi_ref_wf = init_magnitude_wf(2, name="epi_ref_wf")
    epi_ref_wf.inputs.inputnode.magnitude = (
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

    gen_coeff = pe.Node(niu.Function(function=_gen_coeff), name="gen_coeff")

    reg_wf = init_coeff2epi_wf(2, debug=True, write_coeff=True)

    workflow = pe.Workflow(name="test_registration_wf")
    # fmt: off
    workflow.connect([
        (epi_ref_wf, reg_wf, [
            ("outputnode.fmap_ref", "inputnode.target_ref"),
            ("outputnode.fmap_mask", "inputnode.target_mask"),
        ]),
        (fmap_ref_wf, reg_wf, [
            ("outputnode.fmap_ref", "inputnode.fmap_ref"),
            ("outputnode.fmap_mask", "inputnode.fmap_mask"),
        ]),
        (fmap_ref_wf, gen_coeff, [("outputnode.fmap_ref", "img")]),
        (gen_coeff, reg_wf, [("out", "inputnode.fmap_coeff")]),
    ])
    # fmt: on

    if outdir:
        from niworkflows.interfaces import SimpleBeforeAfter
        from ...outputs import DerivativesDataSink

        report = pe.Node(
            SimpleBeforeAfter(
                after_label="Target EPI",
                before_label="B0 Reference",
            ),
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

        # fmt: off
        workflow.connect([
            (fmap_ref_wf, report, [("outputnode.fmap_ref", "before")]),
            (reg_wf, report, [("outputnode.target_ref", "after")]),
            (report, ds_report, [("out_report", "in_file")]),
        ])
        # fmt: on

    if workdir:
        workflow.base_dir = str(workdir)

    workflow.run(plugin="Linear")


# def test_map_coeffs(tmpdir):
#     from pathlib import Path
#     tmpdir.chdir()
#     outnii = nb.load(_move_coeff(
#         str(Path(__file__).parent / "sample-coeff.nii.gz"),
#         [str(Path(__file__).parent / "sample-rigid_xfm.mat")]
#     )[0])
#     vs = nb.affines.voxel_sizes(outnii.affine)
#     assert np.allclose(vs, (40., 40., 20.))

#     dircos = outnii.affine[:3, :3] / vs
#     assert np.allclose(dircos, np.eye(3))


def _gen_coeff(img):
    from pathlib import Path
    from sdcflows.interfaces.bspline import bspline_grid

    out_file = Path("coeff.nii.gz").absolute()
    bspline_grid(img).to_filename(out_file)
    return str(out_file)
