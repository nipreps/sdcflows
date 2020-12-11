"""test_fieldmaps."""
import pytest
from .. import fieldmaps as fm


def test_FieldmapFile(testdata_dir):
    """Test one existing file."""
    fm.FieldmapFile(testdata_dir / "sub-01" / "anat" / "sub-01_T1w.nii.gz")


@pytest.mark.parametrize(
    "inputfiles,method,nsources,raises",
    [
        (
            ("fmap/sub-01_fieldmap.nii.gz", "fmap/sub-01_magnitude.nii.gz"),
            fm.EstimatorType.MAPPED,
            2,
            False,
        ),
        (
            (
                "fmap/sub-01_phase1.nii.gz",
                "fmap/sub-01_phase2.nii.gz",
                "fmap/sub-01_magnitude1.nii.gz",
                "fmap/sub-01_magnitude2.nii.gz",
            ),
            fm.EstimatorType.PHASEDIFF,
            4,
            False,
        ),
        (
            ("fmap/sub-01_phase1.nii.gz", "fmap/sub-01_phase2.nii.gz"),
            fm.EstimatorType.PHASEDIFF,
            4,
            True,
        ),
        (("fmap/sub-01_phase2.nii.gz",), fm.EstimatorType.PHASEDIFF, 4, True),
        (("fmap/sub-01_phase1.nii.gz",), fm.EstimatorType.PHASEDIFF, 4, True),
        (
            ("fmap/sub-01_dir-LR_epi.nii.gz", "fmap/sub-01_dir-RL_epi.nii.gz"),
            fm.EstimatorType.PEPOLAR,
            2,
            False,
        ),
        (
            ("fmap/sub-01_dir-LR_epi.nii.gz", "dwi/sub-01_dir-RL_sbref.nii.gz"),
            fm.EstimatorType.PEPOLAR,
            2,
            False,
        ),
        (
            ("anat/sub-01_T1w.nii.gz", "dwi/sub-01_dir-RL_sbref.nii.gz"),
            fm.EstimatorType.ANAT,
            2,
            False,
        ),
    ],
)
def test_FieldmapEstimation(testdata_dir, inputfiles, method, nsources, raises):
    """Test errors."""
    sub_dir = testdata_dir / "sub-01"

    sources = [sub_dir / f for f in inputfiles]

    if raises is True:
        # Ensure that _estimators is still holding values from previous
        # parameter set of this parametrized execution.
        with pytest.raises(ValueError):
            fm.FieldmapEstimation(sources)

        # Clean up so this parameter set can be tested.
        fm.clear_registry()

    fe = fm.FieldmapEstimation(sources)
    assert fe.method == method
    assert len(fe.sources) == nsources
    assert fe.bids_id is not None and fe.bids_id.startswith("auto_")

    # Attempt to change bids_id
    with pytest.raises(ValueError):
        fe.bids_id = "other"

    # Setting the same value should not raise
    fe.bids_id = fe.bids_id

    # Ensure duplicate B0FieldIdentifier are not accepted
    with pytest.raises(KeyError):
        fm.FieldmapEstimation(sources, bids_id=fe.bids_id)

    # Ensure we can't instantiate one more estimation with same sources
    with pytest.raises(ValueError):
        fm.FieldmapEstimation(sources, bids_id=f"my{fe.bids_id}")

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

    fm.clear_registry()

    with pytest.raises(errortype):
        fm.FieldmapEstimation([sub_dir / f for f in inputfiles])

    fm.clear_registry()


def test_FieldmapEstimationIdentifier(monkeypatch, testdata_dir):
    """Check some use cases of B0FieldIdentifier."""
    fm.clear_registry()

    with pytest.raises(ValueError):
        fm.FieldmapEstimation(
            [
                fm.FieldmapFile(
                    testdata_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                    metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"},
                ),
                fm.FieldmapFile(
                    testdata_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                    metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_1"},
                ),
            ]
        )  # Inconsistent identifiers

    fe = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(
                testdata_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                metadata={
                    "Units": "Hz",
                    "B0FieldIdentifier": "fmap_0",
                    "IntendedFor": "file1.nii.gz",
                },
            ),
            fm.FieldmapFile(
                testdata_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"},
            ),
        ]
    )
    assert fe.bids_id == "fmap_0"
    assert fm.get_identifier("file1.nii.gz") == ("fmap_0",)
    assert not fm.get_identifier("file2.nii.gz")

    with pytest.raises(KeyError):
        fm.FieldmapEstimation(
            [
                fm.FieldmapFile(
                    testdata_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                    metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"},
                ),
                fm.FieldmapFile(
                    testdata_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                    metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"},
                ),
            ]
        )  # Consistent, but already exists

    fm.clear_registry()

    fe = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(
                testdata_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                metadata={
                    "Units": "Hz",
                    "B0FieldIdentifier": "fmap_1",
                    "IntendedFor": ["file1.nii.gz", "file2.nii.gz"],
                },
            ),
            fm.FieldmapFile(
                testdata_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_1"},
            ),
        ]
    )
    assert fe.bids_id == "fmap_1"
    assert fm.get_identifier("file1.nii.gz") == ("fmap_1",)
    assert fm.get_identifier("file2.nii.gz") == ("fmap_1",)
    assert not fm.get_identifier("file3.nii.gz")
    assert fm.get_identifier(
        str(testdata_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz"), by="sources"
    ) == ("fmap_1",)

    with monkeypatch.context() as m:
        m.setattr(fm, "_intents", {"file1.nii.gz": {"fmap_0", "fmap_1"}})
        assert fm.get_identifier("file1.nii.gz") == ("fmap_0", "fmap_1",)

    fm.clear_registry()
