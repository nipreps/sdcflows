import re
from pathlib import Path
from shutil import rmtree

import pytest

from niworkflows.utils.testing import generate_bids_skeleton

from sdcflows.utils.wrangler import find_estimators
from sdcflows.fieldmaps import clear_registry, get_identifier


def gen_layout(bids_dir, database_dir=None):
    import re
    from bids.layout import BIDSLayout, BIDSLayoutIndexer

    _indexer = BIDSLayoutIndexer(
        validate=False,
        ignore=(
            "code",
            "stimuli",
            "sourcedata",
            "models",
            "derivatives",
            re.compile(r"^\."),
            re.compile(r"sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|eeg|ieeg|meg|micr|perf)"),
        ),
    )

    layout_kwargs = {"indexer": _indexer}

    if database_dir:
        layout_kwargs["database_path"] = database_dir

    layout = BIDSLayout(bids_dir, **layout_kwargs)
    return layout


pepolar = {
    "01": [
        {
            "session": "01",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {
                    "suffix": "epi",
                    "dir": "AP",
                    "metadata": {
                        "EchoTime": 1.2,
                        "PhaseEncodingDirection": "j-",
                        "TotalReadoutTime": 0.8,
                        "IntendedFor": "ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz",
                    },
                },
                {
                    "suffix": "epi",
                    "dir": "PA",
                    "metadata": {
                        "EchoTime": 1.2,
                        "PhaseEncodingDirection": "j",
                        "TotalReadoutTime": 0.8,
                        "IntendedFor": "ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz",
                    },
                },
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ],
        },
        {
            "session": "02",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {
                    "suffix": "epi",
                    "dir": "AP",
                    "metadata": {
                        "EchoTime": 1.2,
                        "PhaseEncodingDirection": "j-",
                        "TotalReadoutTime": 0.8,
                        "IntendedFor": (
                            "bids::sub-01/ses-02/func/sub-01_ses-02_task-rest_bold.nii.gz"
                        ),
                    },
                },
                {
                    "suffix": "epi",
                    "dir": "PA",
                    "metadata": {
                        "EchoTime": 1.2,
                        "PhaseEncodingDirection": "j",
                        "TotalReadoutTime": 0.8,
                        "IntendedFor": (
                            "bids::sub-01/ses-02/func/sub-01_ses-02_task-rest_bold.nii.gz"
                        ),
                    },
                },
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ],
        },
        {
            "session": "03",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {
                    "suffix": "epi",
                    "dir": "AP",
                    "metadata": {
                        "EchoTime": 1.2,
                        "PhaseEncodingDirection": "j-",
                        "TotalReadoutTime": 0.8,
                        "IntendedFor": (
                            "bids::sub-01/ses-03/func/sub-01_ses-03_task-rest_bold.nii.gz"
                        ),
                    },
                },
                {
                    "suffix": "epi",
                    "dir": "PA",
                    "metadata": {
                        "EchoTime": 1.2,
                        "PhaseEncodingDirection": "j",
                        "TotalReadoutTime": 0.8,
                        "IntendedFor": (
                            "bids::sub-01/ses-03/func/sub-01_ses-03_task-rest_bold.nii.gz"
                        ),
                    },
                },
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ],
        },
        {
            "session": "04",
            "anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}],
            "fmap": [
                {
                    "suffix": "epi",
                    "dir": "AP",
                    "metadata": {
                        "EchoTime": 1.2,
                        "PhaseEncodingDirection": "j-",
                        "TotalReadoutTime": 0.8,
                        "IntendedFor": [
                            "ses-04/func/sub-01_ses-04_task-rest_run-1_bold.nii.gz",
                            "ses-04/func/sub-01_ses-04_task-rest_run-2_bold.nii.gz",
                        ],
                    },
                },
            ],
            "func": [
                {
                    "task": "rest",
                    "run": 1,
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                },
                {
                    "task": "rest",
                    "run": 2,
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                },
            ],
        },
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
                        "IntendedFor": "ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz",
                    },
                },
                {"suffix": "magnitude1", "metadata": {"EchoTime": 1.2}},
                {"suffix": "magnitude2", "metadata": {"EchoTime": 1.4}},
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ],
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
                        "IntendedFor": (
                            "bids::sub-01/ses-02/func/sub-01_ses-02_task-rest_bold.nii.gz"
                        ),
                    },
                },
                {"suffix": "magnitude1", "metadata": {"EchoTime": 1.2}},
                {"suffix": "magnitude2", "metadata": {"EchoTime": 1.4}},
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ],
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
                        "IntendedFor": (
                            "bids::sub-01/ses-03/func/sub-01_ses-03_task-rest_bold.nii.gz"
                        ),
                    },
                },
                {"suffix": "magnitude1", "metadata": {"EchoTime": 1.2}},
                {"suffix": "magnitude2", "metadata": {"EchoTime": 1.4}},
            ],
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "TotalReadoutTime": 0.5,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ],
        },
    ]
}


filters = {
    "fmap": {
        "datatype": "fmap",
        "session": "01",
    },
    "t1w": {"datatype": "anat", "session": "01", "suffix": "T1w"},
    "bold": {"datatype": "func", "session": "01", "suffix": "bold"},
}


@pytest.mark.parametrize(
    'name,skeleton,estimations',
    [
        ('pepolar', pepolar, 1),
        ('phasediff', phasediff, 1),
    ],
)
def test_wrangler_filter(tmpdir, name, skeleton, estimations):
    bids_dir = str(tmpdir / name)
    generate_bids_skeleton(bids_dir, skeleton)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', bids_filters=filters['fmap'])
    assert len(est) == estimations
    clear_registry()


@pytest.mark.parametrize(
    'name,skeleton,total_estimations',
    [
        ('pepolar', pepolar, 5),
        ('phasediff', phasediff, 3),
    ],
)
@pytest.mark.parametrize("session, estimations", [
    ("01", 1),
    ("02", 1),
    ("03", 1),
    (None, None),
])
def test_wrangler_URIs(tmpdir, name, skeleton, session, estimations, total_estimations):
    bids_dir = str(tmpdir / name)
    generate_bids_skeleton(bids_dir, skeleton)
    layout = gen_layout(bids_dir)
    est = find_estimators(
        layout=layout,
        subject='01',
        sessions=[session] if session else None,
    )
    assert len(est) == estimations or total_estimations
    if session:
        bold = layout.get(session=session, suffix="bold", extension=".nii.gz")[0]
        intended_rel = re.sub(r'^sub-[a-zA-Z0-9]*/', '', str(Path(bold).relative_to(layout.root)))
        b0_id = get_identifier(intended_rel)
        assert b0_id == ('auto_00000',)

    clear_registry()


def test_single_reverse_pedir(tmp_path):
    bids_dir = tmp_path / "bids"
    generate_bids_skeleton(bids_dir, pepolar)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', bids_filters={'session': '04'})
    assert len(est) == 2
    subject_root = bids_dir / 'sub-01'
    for estimator in est:
        assert len(estimator.sources) == 2
        epi, bold = estimator.sources
        # Just checking order
        assert epi.entities['fmap'] == 'epi'
        # IntendedFor is a list of strings
        # REGRESSION: The result was a PyBIDS BIDSFile (fmriprep#3020)
        assert epi.metadata['IntendedFor'] == [str(bold.path.relative_to(subject_root))]


def test_fieldmapless(tmp_path):
    bids_dir = tmp_path / "bids"

    T1w = {"suffix": "T1w"}
    bold = {
        "task": "rest",
        "suffix": "bold",
        "metadata": {
            "RepetitionTime": 0.8,
            "TotalReadoutTime": 0.5,
            "PhaseEncodingDirection": "j",
        },
    }
    me_metadata = [{"EchoTime": 0.01 * i, **bold["metadata"]} for i in range(1, 4)]
    sbref = {**bold, **{"suffix": "sbref"}}
    spec = {
        "01": {
            "anat": [T1w],
            "func": [bold],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject="01", fmapless=True)
    assert len(est) == 1
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Multi-run generates one estimator per-run
    spec = {
        "01": {
            "anat": [T1w],
            "func": [{"run": i, **bold} for i in range(1, 3)],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject="01", fmapless=True)
    assert len(est) == 2
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Multi-echo should only generate one estimator
    spec = {
        "01": {
            "anat": [T1w],
            "func": [{"echo": i + 1, **bold, **{"metadata": me_metadata[i]}} for i in range(3)],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject="01", fmapless=True)
    assert len(est) == 1
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Matching bold+sbref should generate only one estimator
    spec = {
        "01": {
            "anat": [T1w],
            "func": [bold, sbref],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject="01", fmapless=True)
    assert len(est) == 1
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Mismatching bold+sbref should generate two sbrefs
    spec = {
        "01": {
            "anat": [T1w],
            "func": [{"acq": "A", **bold}, {"acq": "B", **sbref}],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject="01", fmapless=True)
    assert len(est) == 2
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Multiecho bold+sbref should generate only one estimator
    spec = {
        "01": {
            "anat": [T1w],
            "func": [{"echo": i + 1, **bold, **{"metadata": me_metadata[i]}} for i in range(3)]
            + [{"echo": i + 1, **sbref, **{"metadata": me_metadata[i]}} for i in range(3)],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject="01", fmapless=True)
    assert len(est) == 1
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)
