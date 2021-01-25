"""Test fieldmap-less SDC-SyN."""
import os
import pytest
from nipype.pipeline import engine as pe
from niworkflows.interfaces.nibabel import ApplyMask

from ..syn import init_syn_sdc_wf


@pytest.mark.skipif(os.getenv("TRAVIS") == "true", reason="this is TravisCI")
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="this is GH Actions")
def test_syn_wf(tmpdir, datadir, workdir, outdir):
    """Build and run an SDC-SyN workflow."""
    derivs_path = datadir / "ds000054" / "derivatives"
    smriprep = derivs_path / "smriprep-0.6" / "sub-100185" / "anat"

    wf = pe.Workflow(name="syn_test")

    syn_wf = init_syn_sdc_wf(debug=True, omp_nthreads=4)
    syn_wf.inputs.inputnode.epi_ref = (
        str(
            derivs_path
            / "sdcflows-tests"
            / "sub-100185_task-machinegame_run-1_boldref.nii.gz"
        ),
        {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.005},
    )
    syn_wf.inputs.inputnode.epi_mask = str(
        derivs_path
        / "sdcflows-tests"
        / "sub-100185_task-machinegame_run-1_desc-brain_mask.nii.gz"
    )
    syn_wf.inputs.inputnode.anat2epi_xfm = str(
        derivs_path / "sdcflows-tests" / "t1w2bold.txt"
    )
    syn_wf.inputs.inputnode.std2anat_xfm = str(
        smriprep / "sub-100185_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
    )

    t1w_mask = pe.Node(
        ApplyMask(
            in_file=str(smriprep / "sub-100185_desc-preproc_T1w.nii.gz"),
            in_mask=str(smriprep / "sub-100185_desc-brain_mask.nii.gz"),
        ),
        name="t1w_mask",
    )

    # fmt: off
    wf.connect([
        (t1w_mask, syn_wf, [("out_file", "inputnode.anat_brain")]),
    ])
    # fmt: on

    if outdir:
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        outdir = outdir / "unittests" / "syn_test"
        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(outdir),
            write_coeff=True,
            bids_fmap_id="sdcsyn",
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = [
            str(
                derivs_path
                / "sdcflows-tests"
                / "sub-100185_task-machinegame_run-1_boldref.nii.gz"
            )
        ]
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = {
            "PhaseEncodingDirection": "j-"
        }

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(outdir),
            fmap_type="sdcsyn",
        )
        fmap_reports_wf.inputs.inputnode.source_files = [
            str(
                derivs_path
                / "sdcflows-tests"
                / "sub-100185_task-machinegame_run-1_boldref.nii.gz"
            )
        ]

        # fmt: off
        wf.connect([
            (syn_wf, fmap_reports_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask")]),
            (syn_wf, fmap_derivatives_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
            ]),
        ])
        # fmt: on

    if workdir:
        wf.base_dir = str(workdir)

    wf.run(plugin="Linear")


@pytest.mark.parametrize("ants_version", ["2.2.0", "2.1.0", None])
def test_syn_wf_version(monkeypatch, ants_version):
    """Ensure errors are triggered with ANTs < 2.2."""
    from niworkflows.interfaces.fixes import FixHeaderRegistration as Registration

    monkeypatch.setattr(Registration, "version", ants_version)
    if ants_version == "2.1.0":
        with pytest.raises(RuntimeError):
            init_syn_sdc_wf(debug=True, omp_nthreads=4)
    else:
        wf = init_syn_sdc_wf(debug=True, omp_nthreads=4)
        assert (ants_version or "version unknown") in wf.__desc__
