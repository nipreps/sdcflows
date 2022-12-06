import pytest

from niworkflows.utils.testing import generate_bids_skeleton

from sdcflows.cli.find_estimators import gen_layout
from sdcflows.utils.wrangler import find_estimators
from sdcflows.fieldmaps import clear_registry


pepolar = {
    "01": [
        {
            "session": "01",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {"suffix": "epi", "dir": "AP", "metadata": {
                    "EchoTime": 1.2,
                    "PhaseEncodingDirection": "j-",
                    "TotalReadoutTime": 0.8,
                    "IntendedFor": "ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz"
                }},
                {"suffix": "epi", "dir": "PA", "metadata": {
                    "EchoTime": 1.2,
                    "PhaseEncodingDirection": "j",
                    "TotalReadoutTime": 0.8,
                    "IntendedFor": "ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz"
                }}
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j"
                    }
                }
            ]
        },
        {
            "session": "02",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {"suffix": "epi", "dir": "AP", "metadata": {
                    "EchoTime": 1.2,
                    "PhaseEncodingDirection": "j-",
                    "TotalReadoutTime": 0.8,
                    "IntendedFor": "ses-02/func/sub-01_ses-02_task-rest_bold.nii.gz"
                }},
                {"suffix": "epi", "dir": "PA", "metadata": {
                    "EchoTime": 1.2,
                    "PhaseEncodingDirection": "j",
                    "TotalReadoutTime": 0.8,
                    "IntendedFor": "ses-02/func/sub-01_ses-02_task-rest_bold.nii.gz"
                }}
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j"
                    }
                }
            ]
        },
        {
            "session": "03",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {"suffix": "epi", "dir": "AP", "metadata": {
                    "EchoTime": 1.2,
                    "PhaseEncodingDirection": "j-",
                    "TotalReadoutTime": 0.8,
                    "IntendedFor": "ses-03/func/sub-01_ses-03_task-rest_bold.nii.gz"
                }},
                {"suffix": "epi", "dir": "PA", "metadata": {
                    "EchoTime": 1.2,
                    "PhaseEncodingDirection": "j",
                    "TotalReadoutTime": 0.8,
                    "IntendedFor": "ses-03/func/sub-01_ses-03_task-rest_bold.nii.gz"
                }}
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j"
                    }
                }
            ]
        }
    ]
}


phasediff = {
    "01": [
        {
            "session": "01",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {
                    "suffix": "phasediff",
                    "metadata": {
                        "EchoTime1": 1.2,
                        "EchoTime2": 1.4,
                        "IntendedFor": "ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz"
                    }
                },
                {"suffix": "magnitude1", "metadata": {"EchoTime": 1.2}},
                {"suffix": "magnitude2", "metadata": {"EchoTime": 1.4}}
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j"
                    }
                }
            ]
        },
        {
            "session": "02",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {
                    "suffix": "phasediff",
                    "metadata": {
                        "EchoTime1": 1.2,
                        "EchoTime2": 1.4,
                        "IntendedFor": "ses-02/func/sub-01_ses-02_task-rest_bold.nii.gz"
                    }
                },
                {"suffix": "magnitude1", "metadata": {"EchoTime": 1.2}},
                {"suffix": "magnitude2", "metadata": {"EchoTime": 1.4}}
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j"
                    }
                }
            ]
        },
        {
            "session": "03",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {
                    "suffix": "phasediff",
                    "metadata": {
                        "EchoTime1": 1.2,
                        "EchoTime2": 1.4,
                        "IntendedFor": "ses-03/func/sub-01_ses-03_task-rest_bold.nii.gz"
                    }
                },
                {"suffix": "magnitude1", "metadata": {"EchoTime": 1.2}},
                {"suffix": "magnitude2", "metadata": {"EchoTime": 1.4}}
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j"
                    }
                }
            ]
        }
    ]
}


filters = {
    "fmap": {
        "datatype": "fmap",
        "session": "01",
    },
    "t1w": {
        "datatype": "anat",
        "session": "01",
        "suffix": "T1w"
    },
    "bold": {
        "datatype": "func",
        "session": "01",
        "suffix": "bold"
    }
}


@pytest.mark.parametrize('name,skeleton,estimations', [
    ('pepolar', pepolar, 1),
    ('phasediff', phasediff, 1),
])
def test_wrangler_filter(tmpdir, name, skeleton, estimations):
    bids_dir = str(tmpdir / name)
    generate_bids_skeleton(bids_dir, skeleton)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', bids_filters=filters['fmap'])
    assert len(est) == estimations
    clear_registry()
