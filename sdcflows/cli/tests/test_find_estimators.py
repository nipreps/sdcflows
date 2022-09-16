import pytest
from niworkflows.utils.testing import generate_bids_skeleton

from ..find_estimators import main as find_estimators
from ...fieldmaps import clear_registry

OUTPUT = """\
Estimation for <{path}> complete. Found:
\tsub-01
\t\tFieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, \
bids_id='{estimator_id}')
\t\t\tj-\tfmap/sub-01_dir-AP_epi.nii.gz
\t\t\tj\tfmap/sub-01_dir-PA_epi.nii.gz
\tsub-02
\t\tNo estimators found
"""

intendedfor_config = {
    "01": [
        {"anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}]},
        {
            "fmap": [
                {
                    "dir": "AP",
                    "suffix": "epi",
                    "metadata": {
                        "TotalReadoutTime": 0.1,
                        "PhaseEncodingDirection": "j-",
                        "IntendedFor": "func/sub-01_task-rest_bold.nii.gz",
                    },
                },
                {
                    "dir": "PA",
                    "suffix": "epi",
                    "metadata": {
                        "TotalReadoutTime": 0.1,
                        "PhaseEncodingDirection": "j",
                        "IntendedFor": "func/sub-01_task-rest_bold.nii.gz",
                    },
                },
            ]
        },
        {
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ]
        },
    ],
    "02": [{"anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}]}],
}


b0field_config = {
    "01": [
        {"anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}]},
        {
            "fmap": [
                {
                    "dir": "AP",
                    "suffix": "epi",
                    "metadata": {
                        "B0FieldIdentifier": "pepolar",
                        "TotalReadoutTime": 0.1,
                        "PhaseEncodingDirection": "j-",
                    },
                },
                {
                    "dir": "PA",
                    "suffix": "epi",
                    "metadata": {
                        "B0FieldIdentifier": "pepolar",
                        "TotalReadoutTime": 0.1,
                        "PhaseEncodingDirection": "j",
                    },
                },
            ]
        },
        {
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "B0FieldSource": "pepolar",
                        "RepetitionTime": 0.8,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ]
        },
    ],
    "02": [{"anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}]}],
}


@pytest.mark.parametrize(
    "test_id,config,estimator_id",
    [("intendedfor", intendedfor_config, "auto_00000"), ("b0field", b0field_config, "pepolar")],
)
def test_find_estimators(tmp_path, capsys, test_id, config, estimator_id):
    path = tmp_path / test_id
    generate_bids_skeleton(path, config)
    find_estimators([str(path)])
    output = OUTPUT.format(path=path, estimator_id=estimator_id)
    out, _ = capsys.readouterr()
    assert out == output
    clear_registry()
