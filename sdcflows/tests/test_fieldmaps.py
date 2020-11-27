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

    sources = [sub_dir / f for f in inputfiles]
    fe = FieldmapEstimation(sources)
    assert fe.method == method
    assert len(fe.sources) == nsources
    assert fe.bids_id is not None and fe.bids_id.startswith("auto_")

    # Attempt to change bids_id
    with pytest.raises(ValueError):
        fe.bids_id = "other"

    # Setting the same value should not raise
    fe.bids_id = fe.bids_id

    # Ensure duplicate B0FieldIdentifier are not accepted
    with pytest.raises(ValueError):
        FieldmapEstimation(sources, bids_id=fe.bids_id)

    # B0FieldIdentifier can be generated manually
    # Creating two FieldmapEstimation objects from the same sources SHOULD fail
    # or be better handled in the future (see #129).
    fe2 = FieldmapEstimation(sources, bids_id=f"no{fe.bids_id}")
    assert fe2.bids_id and fe2.bids_id.startswith("noauto_")

    # Exercise workflow creation
    wf = fe.get_workflow()
    wf == fe.get_workflow()


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


def test_FieldmapEstimationIdentifier(testdata_dir):
    """Check some use cases of B0FieldIdentifier."""
    with pytest.raises(ValueError):
        FieldmapEstimation([
            FieldmapFile(testdata_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                         metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"}),
            FieldmapFile(testdata_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                         metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_1"})
        ])  # Inconsistent identifiers

    fe = FieldmapEstimation([
        FieldmapFile(testdata_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                     metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"}),
        FieldmapFile(testdata_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                     metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"})
    ])
    assert fe.bids_id == "fmap_0"

    with pytest.raises(ValueError):
        FieldmapEstimation([
            FieldmapFile(testdata_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                         metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"}),
            FieldmapFile(testdata_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                         metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"})
        ])  # Consistent, but already exists

    fe = FieldmapEstimation([
        FieldmapFile(testdata_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                     metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_1"}),
        FieldmapFile(testdata_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                     metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_1"})
    ])
    assert fe.bids_id == "fmap_1"
