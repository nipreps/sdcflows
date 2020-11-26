"""Test fieldmap interfaces."""
import numpy as np
import nibabel as nb
import pytest

from ..fmap import CheckB0Units


@pytest.mark.parametrize("units", ("rad/s", "Hz"))
def test_units(tmpdir, units):
    """Check the conversion of units."""
    tmpdir.chdir()
    hz = np.ones((5, 5, 5), dtype="float32") * 100
    data = hz.copy()

    if units == "rad/s":
        data *= 2.0 * np.pi

    nb.Nifti1Image(data, np.eye(4), None).to_filename("data.nii.gz")
    out_data = nb.load(
        CheckB0Units(units=units, in_file="data.nii.gz").run().outputs.out_file
    ).get_fdata(dtype="float32")

    assert np.allclose(hz, out_data)
