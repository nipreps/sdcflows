"""test_fieldmaps."""
import pytest
from ..fieldmaps import FieldmapFile, FieldmapEstimation, EstimatorType


def test_FieldmapFile(testdata_dir):
    """Test one existing file."""
    FieldmapFile(testdata_dir / "sub-01" / "anat" / "sub-01_T1w.nii.gz")


@pytest.mark.parametrize(
    "inputfiles,method,nsources",
    [
        (
            ("fmap/sub-01_fieldmap.nii.gz", "fmap/sub-01_magnitude.nii.gz"),
            EstimatorType.MAPPED,
            2,
        ),
        (
            (
                "fmap/sub-01_phase1.nii.gz",
                "fmap/sub-01_phase2.nii.gz",
                "fmap/sub-01_magnitude1.nii.gz",
                "fmap/sub-01_magnitude2.nii.gz",
            ),
            EstimatorType.PHASEDIFF,
            4,
        ),
        (
            ("fmap/sub-01_phase1.nii.gz", "fmap/sub-01_phase2.nii.gz"),
            EstimatorType.PHASEDIFF,
            4,
        ),
        (("fmap/sub-01_phase2.nii.gz",), EstimatorType.PHASEDIFF, 4),
        (("fmap/sub-01_phase1.nii.gz",), EstimatorType.PHASEDIFF, 4),
        (
            ("fmap/sub-01_dir-LR_epi.nii.gz", "fmap/sub-01_dir-RL_epi.nii.gz"),
            EstimatorType.PEPOLAR,
            2,
        ),
        (
            ("fmap/sub-01_dir-LR_epi.nii.gz", "dwi/sub-01_dir-RL_sbref.nii.gz"),
            EstimatorType.PEPOLAR,
            2,
        ),
        (
            ("anat/sub-01_T1w.nii.gz", "dwi/sub-01_dir-RL_sbref.nii.gz"),
            EstimatorType.ANAT,
            2,
        ),
    ],
)
def test_FieldmapEstimation(testdata_dir, inputfiles, method, nsources):
    """Test errors."""
    sub_dir = testdata_dir / "sub-01"

    fe = FieldmapEstimation([sub_dir / f for f in inputfiles])
    assert fe.method == method
    assert len(fe.sources) == nsources


@pytest.mark.parametrize(
    "inputfiles,errortype",
    [
        (("fmap/sub-01_fieldmap.nii.gz", "fmap/sub-01_phasediff.nii.gz"), TypeError),
        (("fmap/sub-01_dir-RL_epi.nii.gz",), ValueError),
        (("anat/sub-01_T1w.nii.gz",), ValueError),
        (("anat/sub-01_T1w.nii.gz", "fmap/sub-01_phase2.nii.gz"), TypeError),
    ],
)
def test_FieldmapEstimationError(testdata_dir, inputfiles, errortype):
    """Test errors."""
    sub_dir = testdata_dir / "sub-01"

    with pytest.raises(errortype):
        FieldmapEstimation([sub_dir / f for f in inputfiles])
