"""Test phase manipulation routines."""
import numpy as np
import nibabel as nb

from ..phasemanip import au2rads, phdiff2fmap


def test_au2rads(tmp_path):
    """Check the conversion."""
    data = np.random.randint(0, high=4096, size=(5, 5, 5))
    data[0, 0, 0] = 0
    data[-1, -1, -1] = 4096

    nb.Nifti1Image(data.astype("int16"), np.eye(4)).to_filename(
        tmp_path / "testdata.nii.gz"
    )

    out_file = au2rads(tmp_path / "testdata.nii.gz")

    assert np.allclose(
        (data / 4096).astype("float32") * 2.0 * np.pi,
        nb.load(out_file).get_fdata(dtype="float32"),
    )


def test_phdiff2fmap(tmp_path):
    """Check the conversion."""
    nb.Nifti1Image(
        np.ones((5, 5, 5), dtype="float32") * 2.0 * np.pi * 2.46e-3, np.eye(4)
    ).to_filename(tmp_path / "testdata.nii.gz")

    out_file = phdiff2fmap(tmp_path / "testdata.nii.gz", 2.46e-3)

    assert np.allclose(np.ones((5, 5, 5)), nb.load(out_file).get_fdata(dtype="float32"))
