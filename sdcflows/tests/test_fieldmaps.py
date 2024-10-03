# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""test_fieldmaps."""
from collections import namedtuple
import shutil
import pytest
import bids
from .. import fieldmaps as fm


@pytest.fixture(autouse=True)
def clear_registry():
    fm.clear_registry()
    yield
    fm.clear_registry()


def test_FieldmapFile(dsA_dir):
    """Test one existing file."""
    f1 = fm.FieldmapFile(dsA_dir / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    f2 = fm.FieldmapFile(str(dsA_dir / "sub-01" / "anat" / "sub-01_T1w.nii.gz"))
    f3 = fm.FieldmapFile(
        bids.layout.BIDSFile(str(dsA_dir / "sub-01" / "anat" / "sub-01_T1w.nii.gz"))
    )
    assert f1 == f2 == f3

    with pytest.raises(ValueError):
        fm.FieldmapFile(dsA_dir / "sub-01" / "fmap" / "sub-01_dir-AP_epi.json")

    with pytest.raises(ValueError):
        fm.FieldmapFile(dsA_dir / "sub-01" / "anat" / "sub-01_FLAIR.nii.gz")


@pytest.mark.parametrize(
    "inputfiles,method,nsources",
    [
        (
            ("fmap/sub-01_fieldmap.nii.gz", "fmap/sub-01_magnitude.nii.gz"),
            fm.EstimatorType.MAPPED,
            2,
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
        ),
        (
            ("fmap/sub-01_phase1.nii.gz", "fmap/sub-01_phase2.nii.gz"),
            fm.EstimatorType.PHASEDIFF,
            4,
        ),
        (("fmap/sub-01_phase2.nii.gz",), fm.EstimatorType.PHASEDIFF, 4),
        (("fmap/sub-01_phase1.nii.gz",), fm.EstimatorType.PHASEDIFF, 4),
        (
            ("fmap/sub-01_dir-LR_epi.nii.gz", "fmap/sub-01_dir-RL_epi.nii.gz"),
            fm.EstimatorType.PEPOLAR,
            2,
        ),
        (
            ("fmap/sub-01_dir-LR_epi.nii.gz", "dwi/sub-01_dir-RL_sbref.nii.gz"),
            fm.EstimatorType.PEPOLAR,
            2,
        ),
        (
            ("anat/sub-01_T1w.nii.gz", "dwi/sub-01_dir-RL_sbref.nii.gz"),
            fm.EstimatorType.ANAT,
            2,
        ),
    ],
)
def test_FieldmapEstimation(dsA_dir, inputfiles, method, nsources):
    """Test errors."""
    sub_dir = dsA_dir / "sub-01"

    sources = [sub_dir / f for f in inputfiles]

    fe = fm.FieldmapEstimation(sources)
    assert fe.method == method
    assert len(fe.sources) == nsources
    assert fe.bids_id is not None and fe.bids_id.startswith("auto_")
    assert fe.bids_id == fe.sanitized_id  # Auto-generated IDs are sanitized

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
def test_FieldmapEstimationError(dsA_dir, inputfiles, errortype):
    """Test errors."""
    sub_dir = dsA_dir / "sub-01"

    fm.clear_registry()

    with pytest.raises(errortype):
        fm.FieldmapEstimation([sub_dir / f for f in inputfiles])

    fm.clear_registry()


def test_FieldmapEstimationIdentifier(monkeypatch, dsA_dir):
    """Check some use cases of B0FieldIdentifier."""
    fm.clear_registry()

    with pytest.raises(ValueError):
        fm.FieldmapEstimation(
            [
                fm.FieldmapFile(
                    dsA_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                    metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"},
                ),
                fm.FieldmapFile(
                    dsA_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                    metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_1"},
                ),
            ]
        )  # Inconsistent identifiers

    fe = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(
                dsA_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                metadata={
                    "Units": "Hz",
                    "B0FieldIdentifier": "fmap_0",
                    "IntendedFor": "file1.nii.gz",
                },
            ),
            fm.FieldmapFile(
                dsA_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
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
                    dsA_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                    metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"},
                ),
                fm.FieldmapFile(
                    dsA_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                    metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_0"},
                ),
            ]
        )  # Consistent, but already exists

    fm.clear_registry()

    fe = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(
                dsA_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                metadata={
                    "Units": "Hz",
                    "B0FieldIdentifier": "fmap_1",
                    "IntendedFor": ["file1.nii.gz", "file2.nii.gz"],
                },
            ),
            fm.FieldmapFile(
                dsA_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                metadata={"Units": "Hz", "B0FieldIdentifier": "fmap_1"},
            ),
        ]
    )
    assert fe.bids_id == "fmap_1"
    assert fm.get_identifier("file1.nii.gz") == ("fmap_1",)
    assert fm.get_identifier("file2.nii.gz") == ("fmap_1",)
    assert not fm.get_identifier("file3.nii.gz")
    assert fm.get_identifier(
        str(dsA_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz"), by="sources"
    ) == ("fmap_1",)

    with monkeypatch.context() as m:
        m.setattr(fm, "_intents", {"file1.nii.gz": {"fmap_0", "fmap_1"}})
        assert fm.get_identifier("file1.nii.gz") == (
            "fmap_0",
            "fmap_1",
        )

    with pytest.raises(KeyError):
        fm.get_identifier("file", by="invalid")

    fm.clear_registry()

    fe = fm.FieldmapEstimation(
        [
            fm.FieldmapFile(
                dsA_dir / "sub-01" / "fmap/sub-01_fieldmap.nii.gz",
                metadata={
                    "Units": "Hz",
                    "B0FieldIdentifier": "fmap-with^special#chars",
                    "IntendedFor": ["file1.nii.gz", "file2.nii.gz"],
                },
            ),
            fm.FieldmapFile(
                dsA_dir / "sub-01" / "fmap/sub-01_magnitude.nii.gz",
                metadata={"Units": "Hz", "B0FieldIdentifier": "fmap-with^special#chars"},
            ),
        ]
    )
    assert fe.bids_id == "fmap-with^special#chars"
    assert fe.sanitized_id == "fmap_with_special_chars"
    # The unsanitized ID is used for lookups
    assert fm.get_identifier("file1.nii.gz") == ("fmap-with^special#chars",)
    assert fm.get_identifier("file2.nii.gz") == ("fmap-with^special#chars",)

    wf = fe.get_workflow()
    assert wf.name == "wf_fmap_with_special_chars"

    fm.clear_registry()


def test_type_setter():
    """Cover the _type_setter routine."""
    obj = namedtuple("FieldmapEstimation", ("method",))(method=fm.EstimatorType.UNKNOWN)
    with pytest.raises(ValueError):
        fm._type_setter(obj, "method", 10)

    obj = namedtuple("FieldmapEstimation", ("method",))(method=fm.EstimatorType.PEPOLAR)
    assert (
        fm._type_setter(obj, "method", fm.EstimatorType.PEPOLAR)
        == fm.EstimatorType.PEPOLAR
    )


def test_FieldmapEstimation_missing_files(tmpdir, dsA_dir):
    """Exercise some FieldmapEstimation checks."""
    tmpdir.chdir()
    tmpdir.mkdir("data")

    # fieldmap - no magnitude
    path = dsA_dir / "sub-01" / "fmap" / "sub-01_fieldmap.nii.gz"
    shutil.copy(path, f"data/{path.name}")

    with pytest.raises(
        ValueError,
        match=r"A fieldmap or phase-difference .* \(magnitude file\) is missing.*",
    ):
        fm.FieldmapEstimation(
            [fm.FieldmapFile("data/sub-01_fieldmap.nii.gz", metadata={"Units": "Hz"})]
        )

    # phase1/2 - no magnitude2
    path = dsA_dir / "sub-01" / "fmap" / "sub-01_phase1.nii.gz"
    shutil.copy(path, f"data/{path.name}")

    path = dsA_dir / "sub-01" / "fmap" / "sub-01_magnitude1.nii.gz"
    shutil.copy(path, f"data/{path.name}")

    path = dsA_dir / "sub-01" / "fmap" / "sub-01_phase2.nii.gz"
    shutil.copy(path, f"data/{path.name}")

    with pytest.raises(
        ValueError, match=r".* \(phase1/2\) .* \(magnitude2 file\) is missing.*"
    ):
        fm.FieldmapEstimation(
            [
                fm.FieldmapFile(
                    "data/sub-01_phase1.nii.gz", metadata={"EchoTime": 0.004}
                ),
                fm.FieldmapFile(
                    "data/sub-01_phase2.nii.gz", metadata={"EchoTime": 0.007}
                ),
            ]
        )

    # pepolar - only one PE
    path = dsA_dir / "sub-01" / "fmap" / "sub-01_dir-AP_epi.nii.gz"
    shutil.copy(path, f"data/{path.name}")

    path = dsA_dir / "sub-01" / "dwi" / "sub-01_dir-AP_dwi.nii.gz"
    shutil.copy(path, f"data/{path.name}")
    with pytest.raises(ValueError, match="Only one phase-encoding direction <j>.*"):
        fm.FieldmapEstimation(
            [
                fm.FieldmapFile(
                    "data/sub-01_dir-AP_epi.nii.gz",
                    metadata={"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.004},
                ),
                fm.FieldmapFile(
                    "data/sub-01_dir-AP_dwi.nii.gz",
                    metadata={"PhaseEncodingDirection": "j", "TotalReadoutTime": 0.004},
                ),
            ]
        )


def test_FieldmapFile_filename(tmp_path, dsA_dir):
    datadir = tmp_path / "phasediff"
    datadir.mkdir(exist_ok=True)

    fmap_path = dsA_dir / "sub-01" / "fmap"
    for fl in fmap_path.glob("*"):
        base = fl.name
        if 'magnitude1' in base or 'magnitude2' in base or 'phasediff' in base:
            print(fl.absolute(), str(datadir / base))

            shutil.copy(fl.absolute(), str(datadir / base))

    # Ensure the correct linked files (magnitude 1/2) are found
    fm.FieldmapEstimation(
        fm.FieldmapFile(datadir / "sub-01_phasediff.nii.gz")
    )
    fm.clear_registry()
