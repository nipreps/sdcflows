"""Test EPI manipulation routines."""
import numpy as np
import nibabel as nb
from ..epimanip import epi_mask


def test_epi_mask(tmpdir, testdata_dir):
    """Check mask algorithm."""
    tmpdir.chdir()
    mask = epi_mask(testdata_dir / "epi.nii.gz")
    assert abs(np.asanyarray(nb.load(mask).dataobj).sum() - 189052) < 10
