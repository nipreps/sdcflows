"""Test utilites."""
import pytest
import numpy as np
import nibabel as nb

from ..utils import Flatten, ConvertWarp


def test_Flatten(tmpdir):
    """Test the flattening interface."""
    tmpdir.chdir()
    shape = (5, 5, 5)
    nb.Nifti1Image(np.zeros(shape), np.eye(4), None).to_filename("file1.nii.gz")
    nb.Nifti1Image(np.zeros((*shape, 6)), np.eye(4), None).to_filename("file2.nii.gz")
    nb.Nifti1Image(np.zeros((*shape, 2)), np.eye(4), None).to_filename("file3.nii.gz")

    out = Flatten(
        in_data=["file1.nii.gz", "file2.nii.gz", "file3.nii.gz"],
        in_meta=[{"a": 1}, {"b": 2}, {"c": 3}],
        max_trs=3,
    ).run()

    assert len(out.outputs.out_list) == 6

    out_meta = out.outputs.out_meta
    assert out_meta[0] == {"a": 1}
    assert out_meta[1] == out_meta[2] == out_meta[3] == {"b": 2}
    assert out_meta[4] == out_meta[5] == {"c": 3}


@pytest.mark.parametrize("shape", [(10, 10, 10, 1, 3), (10, 10, 10, 3)])
def test_ConvertWarp(tmpdir, shape):
    """Exercise the interface."""
    tmpdir.chdir()

    nb.Nifti1Image(np.zeros(shape, dtype="uint8"), np.eye(4), None).to_filename(
        "3dQwarp.nii.gz"
    )

    out = ConvertWarp(in_file="3dQwarp.nii.gz").run()

    nii = nb.load(out.outputs.out_file)
    assert nii.header.get_data_dtype() == np.float32
    assert nii.header.get_intent() == ("vector", (), "")
    assert nii.shape == (10, 10, 10, 1, 3)
