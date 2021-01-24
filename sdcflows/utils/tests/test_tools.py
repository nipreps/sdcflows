"""Test EPI manipulation routines."""
import numpy as np
import nibabel as nb
from ..tools import brain_masker


def test_epi_mask(tmpdir, testdata_dir):
    """Check mask algorithm."""
    tmpdir.chdir()
    mask = brain_masker(testdata_dir / "epi.nii.gz")[-1]
    assert abs(np.asanyarray(nb.load(mask).dataobj).sum() - 166476) < 10
