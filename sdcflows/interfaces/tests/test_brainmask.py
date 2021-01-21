"""Check the tools submodule."""
import os
import pytest
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.ants import N4BiasFieldCorrection
from niworkflows.interfaces.masks import SimpleShowMaskRPT
from ..brainmask import BrainExtraction


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="this is GH Actions")
@pytest.mark.parametrize("folder", ["magnitude/ds000054", "magnitude/ds000217"])
def test_brainmasker(tmpdir, datadir, workdir, outdir, folder):
    """Exercise the brain masking tool."""
    tmpdir.chdir()

    wf = pe.Workflow(name=f"test_mask_{folder.replace('/', '_')}")
    if workdir:
        wf.base_dir = str(workdir)

    input_files = [
        str(f) for f in (datadir / "brain-extraction-tests" / folder).glob("*.nii.gz")
    ]

    inputnode = pe.Node(niu.IdentityInterface(fields=("in_file",)), name="inputnode")
    inputnode.iterables = ("in_file", input_files)
    clipper = pe.Node(niu.Function(function=_advanced_clip), name="clipper")
    n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            shrink_factor=4,
        ),
        n_procs=8,
        name="n4",
    )
    clipper_n4 = pe.Node(
        niu.Function(
            function=_advanced_clip, input_names=["in_file", "p_min", "p_max"]
        ),
        name="clipper_n4",
    )
    clipper_n4.inputs.p_max = 100.0

    masker = pe.Node(BrainExtraction(), name="masker")

    # fmt:off
    wf.connect([
        (inputnode, clipper, [("in_file", "in_file")]),
        (clipper, n4, [("out", "input_image")]),
        (n4, clipper_n4, [("output_image", "in_file")]),
        (clipper_n4, masker, [("out", "in_file")]),
    ])
    # fmt:on

    if outdir:
        out_path = outdir / "masks" / folder.split("/")[-1]
        out_path.mkdir(exist_ok=True, parents=True)
        report = pe.Node(SimpleShowMaskRPT(), name="report")

        def _report_name(fname, out_path):
            from pathlib import Path

            return str(
                out_path
                / Path(fname)
                .name.replace(".nii", "_mask.svg")
                .replace("_magnitude", "_desc-magnitude")
                .replace(".gz", "")
            )

        # fmt: off
        wf.connect([
            (inputnode, report, [(("in_file", _report_name, out_path), "out_report")]),
            (masker, report, [("out_mask", "mask_file")]),
            (clipper_n4, report, [("out", "background_file")]),
        ])
        # fmt: on

    wf.run()


def _advanced_clip(in_file, p_min=35, p_max=99.98, nonzero=True):
    from pathlib import Path
    import nibabel as nb
    import numpy as np

    out_file = Path("clipped.nii.gz").absolute()

    # Load data
    img = nb.load(in_file)
    data = img.get_fdata(dtype="float32")

    # Clip and cast
    a_min = np.percentile(data, p_min)
    if nonzero:
        a_min = max(a_min, 0.0)

    data = np.clip(data, a_min=a_min, a_max=np.percentile(data, p_max))
    data -= data.min()
    data = np.round(data * (255 / data.max())).astype("int16")
    hdr = img.header.copy()
    hdr.set_data_dtype("int16")

    img.__class__(data, img.affine, hdr).to_filename(out_file)

    return str(out_file)
