import re
from pathlib import Path
from shutil import rmtree

import pytest
from niworkflows.utils.testing import generate_bids_skeleton

from sdcflows.fieldmaps import clear_registry, get_identifier
from sdcflows.utils.wrangler import find_estimators


def gen_layout(bids_dir, database_dir=None):
    import re

    from bids.layout import BIDSLayout, BIDSLayoutIndexer

    _indexer = BIDSLayoutIndexer(
        validate=False,
        ignore=(
            'code',
            'stimuli',
            'sourcedata',
            'models',
            'derivatives',
            re.compile(r'^\.'),
            re.compile(r'sub-[a-zA-Z0-9]+(/ses-[a-zA-Z0-9]+)?/(beh|eeg|ieeg|meg|micr|perf)'),
        ),
    )

    layout_kwargs = {'indexer': _indexer}

    if database_dir:
        layout_kwargs['database_path'] = database_dir

    layout = BIDSLayout(bids_dir, **layout_kwargs)
    return layout


pepolar = {
    '01': [
        {
            'session': '01',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'fmap': [
                {
                    'suffix': 'epi',
                    'dir': 'AP',
                    'metadata': {
                        'EchoTime': 1.2,
                        'PhaseEncodingDirection': 'j-',
                        'TotalReadoutTime': 0.8,
                        'IntendedFor': 'ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz',
                    },
                },
                {
                    'suffix': 'epi',
                    'dir': 'PA',
                    'metadata': {
                        'EchoTime': 1.2,
                        'PhaseEncodingDirection': 'j',
                        'TotalReadoutTime': 0.8,
                        'IntendedFor': 'ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz',
                    },
                },
            ],
            'func': [
                {
                    'task': 'rest',
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                }
            ],
        },
        {
            'session': '02',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'fmap': [
                {
                    'suffix': 'epi',
                    'dir': 'AP',
                    'metadata': {
                        'EchoTime': 1.2,
                        'PhaseEncodingDirection': 'j-',
                        'TotalReadoutTime': 0.8,
                        'IntendedFor': (
                            'bids::sub-01/ses-02/func/sub-01_ses-02_task-rest_bold.nii.gz'
                        ),
                    },
                },
                {
                    'suffix': 'epi',
                    'dir': 'PA',
                    'metadata': {
                        'EchoTime': 1.2,
                        'PhaseEncodingDirection': 'j',
                        'TotalReadoutTime': 0.8,
                        'IntendedFor': (
                            'bids::sub-01/ses-02/func/sub-01_ses-02_task-rest_bold.nii.gz'
                        ),
                    },
                },
            ],
            'func': [
                {
                    'task': 'rest',
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                }
            ],
        },
        {
            'session': '03',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'fmap': [
                {
                    'suffix': 'epi',
                    'dir': 'AP',
                    'metadata': {
                        'EchoTime': 1.2,
                        'PhaseEncodingDirection': 'j-',
                        'TotalReadoutTime': 0.8,
                        'IntendedFor': (
                            'bids::sub-01/ses-03/func/sub-01_ses-03_task-rest_bold.nii.gz'
                        ),
                    },
                },
                {
                    'suffix': 'epi',
                    'dir': 'PA',
                    'metadata': {
                        'EchoTime': 1.2,
                        'PhaseEncodingDirection': 'j',
                        'TotalReadoutTime': 0.8,
                        'IntendedFor': (
                            'bids::sub-01/ses-03/func/sub-01_ses-03_task-rest_bold.nii.gz'
                        ),
                    },
                },
            ],
            'func': [
                {
                    'task': 'rest',
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                }
            ],
        },
        {
            'session': '04',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'fmap': [
                {
                    'suffix': 'epi',
                    'dir': 'AP',
                    'metadata': {
                        'EchoTime': 1.2,
                        'PhaseEncodingDirection': 'j-',
                        'TotalReadoutTime': 0.8,
                        'IntendedFor': [
                            'ses-04/func/sub-01_ses-04_task-rest_run-1_bold.nii.gz',
                            'ses-04/func/sub-01_ses-04_task-rest_run-2_bold.nii.gz',
                        ],
                    },
                },
            ],
            'func': [
                {
                    'task': 'rest',
                    'run': 1,
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
                {
                    'task': 'rest',
                    'run': 2,
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                },
            ],
        },
    ]
}


pepolar_b0ids = {
    '01': [
        {
            'session': '01',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'func': [
                {
                    'task': 'rest',
                    'run': 1,
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                        'B0FieldIdentifier': 'b0_pepolar',
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
                {
                    'task': 'rest',
                    'run': 2,
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j-',
                        'B0FieldIdentifier': 'b0_pepolar',
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
            ],
        },
        {
            'session': '02',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'func': [
                {
                    'task': 'rest',
                    'run': 1,
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
                {
                    'task': 'rest',
                    'run': 2,
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j-',
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
                {
                    'task': 'rest',
                    'run': 1,
                    'suffix': 'sbref',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                        'B0FieldIdentifier': 'b0_pepolar',
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
                {
                    'task': 'rest',
                    'run': 2,
                    'suffix': 'sbref',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j-',
                        'B0FieldIdentifier': 'b0_pepolar',
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
            ],
        },
        {
            'session': '03',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'func': [
                {
                    'task': 'rest',
                    'run': 1,
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
                {
                    'task': 'rest',
                    'run': 2,
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j-',
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
                {
                    'task': 'rest',
                    'run': 1,
                    'suffix': 'sbref',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                        'B0FieldIdentifier': ['b0_pepolar', 'b0_pepolar_dup'],
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
                {
                    'task': 'rest',
                    'run': 2,
                    'suffix': 'sbref',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j-',
                        'B0FieldIdentifier': ['b0_pepolar', 'b0_pepolar_dup'],
                        'B0FieldSource': 'b0_pepolar',
                    },
                },
            ],
        },
    ]
}


phasediff = {
    '01': [
        {
            'session': '01',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'fmap': [
                {
                    'suffix': 'phasediff',
                    'metadata': {
                        'EchoTime1': 1.2,
                        'EchoTime2': 1.4,
                        'IntendedFor': 'ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz',
                    },
                },
                {'suffix': 'magnitude1', 'metadata': {'EchoTime': 1.2}},
                {'suffix': 'magnitude2', 'metadata': {'EchoTime': 1.4}},
            ],
            'func': [
                {
                    'task': 'rest',
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                }
            ],
        },
        {
            'session': '02',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'fmap': [
                {
                    'suffix': 'phasediff',
                    'metadata': {
                        'EchoTime1': 1.2,
                        'EchoTime2': 1.4,
                        'IntendedFor': (
                            'bids::sub-01/ses-02/func/sub-01_ses-02_task-rest_bold.nii.gz'
                        ),
                    },
                },
                {'suffix': 'magnitude1', 'metadata': {'EchoTime': 1.2}},
                {'suffix': 'magnitude2', 'metadata': {'EchoTime': 1.4}},
            ],
            'func': [
                {
                    'task': 'rest',
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                }
            ],
        },
        {
            'session': '03',
            'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
            'fmap': [
                {
                    'suffix': 'phasediff',
                    'metadata': {
                        'EchoTime1': 1.2,
                        'EchoTime2': 1.4,
                        'IntendedFor': (
                            'bids::sub-01/ses-03/func/sub-01_ses-03_task-rest_bold.nii.gz'
                        ),
                    },
                },
                {'suffix': 'magnitude1', 'metadata': {'EchoTime': 1.2}},
                {'suffix': 'magnitude2', 'metadata': {'EchoTime': 1.4}},
            ],
            'func': [
                {
                    'task': 'rest',
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 0.8,
                        'TotalReadoutTime': 0.5,
                        'PhaseEncodingDirection': 'j',
                    },
                }
            ],
        },
    ]
}


def _build_medic_skeleton(*, intent: str | None = 'intended_for'):
    """Generate a 3-session × 3-echo × {mag,phase} BIDS skeleton for MEDIC.

    MEDIC is discovered only from BIDS intent metadata, never from file
    structure alone. ``intent`` selects how that intent is expressed on each
    complex BOLD:

    * ``"b0_identifier"`` — the BIDS-RECOMMENDED route: a self-referential
      ``B0FieldIdentifier``/``B0FieldSource`` (the pattern BIDS endorses for
      images that estimate their own B0 field).
    * ``"intended_for"`` — the legacy route: ``IntendedFor`` listing the run's
      6 mag/phase siblings.
    * ``None`` — no intent metadata at all, to confirm MEDIC does **not** fire
      on structure alone.
    """
    echo_times = {'1': 0.0142, '2': 0.03893, '3': 0.06366}
    sessions = []
    for ses in ('01', '02', '03'):
        intended_for = [
            (
                f'bids::sub-01/ses-{ses}/func/'
                f'sub-01_ses-{ses}_task-rest_echo-{echo}_part-{part}_bold.nii.gz'
            )
            for echo in echo_times
            for part in ('mag', 'phase')
        ]
        func = []
        for echo, te in echo_times.items():
            for part in ('mag', 'phase'):
                metadata = {
                    'EchoTime': te,
                    'RepetitionTime': 0.8,
                    'TotalReadoutTime': 0.5,
                    'PhaseEncodingDirection': 'j',
                }
                if intent == 'intended_for':
                    metadata['IntendedFor'] = intended_for
                elif intent == 'b0_identifier':
                    # Every mag+phase echo is an *input* to the estimation, so
                    # all carry B0FieldIdentifier. Only the magnitude echoes are
                    # *corrected* analysis targets, so B0FieldSource sits on mag
                    # alone (the pepolar-style self-correction pattern).
                    b0_id = f'medic_ses{ses}'
                    metadata['B0FieldIdentifier'] = b0_id
                    if part == 'mag':
                        metadata['B0FieldSource'] = b0_id
                func.append(
                    {
                        'task': 'rest',
                        'echo': echo,
                        'part': part,
                        'suffix': 'bold',
                        'metadata': metadata,
                    }
                )
        sessions.append(
            {
                'session': ses,
                'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
                'func': func,
            }
        )
    return {'01': sessions}


medic = _build_medic_skeleton(intent='intended_for')
medic_b0_identifier = _build_medic_skeleton(intent='b0_identifier')
medic_no_intent = _build_medic_skeleton(intent=None)


filters = {
    'fmap': {
        'datatype': 'fmap',
        'session': '01',
    },
    't1w': {'datatype': 'anat', 'session': '01', 'suffix': 'T1w'},
    'bold': {'datatype': 'func', 'session': '01', 'suffix': 'bold'},
    'medic': {'datatype': ['fmap', 'func'], 'session': '01'},
}


@pytest.mark.parametrize(
    'name,skeleton,estimations,bids_filters',
    [
        ('pepolar', pepolar, 1, 'fmap'),
        ('pepolar_b0ids', pepolar_b0ids, 1, 'bold'),
        ('phasediff', phasediff, 1, 'fmap'),
        ('medic', medic, 1, 'medic'),
    ],
)
def test_wrangler_filter(tmpdir, name, skeleton, estimations, bids_filters):
    bids_dir = str(tmpdir / name)
    generate_bids_skeleton(bids_dir, skeleton)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', bids_filters=filters[bids_filters])
    assert len(est) == estimations
    clear_registry()


@pytest.mark.parametrize(
    'name,skeleton,total_estimations,test_auto',
    [
        ('pepolar', pepolar, 5, True),
        ('pepolar_b0ids', pepolar_b0ids, 2, False),
        ('phasediff', phasediff, 3, True),
        ('medic', medic, 3, False),
    ],
)
@pytest.mark.parametrize(
    'session, estimations',
    [
        ('01', 1),
        ('02', 1),
        ('03', 1),
        (None, None),
    ],
)
def test_wrangler_URIs(tmpdir, name, skeleton, session, estimations, total_estimations, test_auto):
    bids_dir = str(tmpdir / name)
    generate_bids_skeleton(bids_dir, skeleton)
    layout = gen_layout(bids_dir)
    est = find_estimators(
        layout=layout,
        subject='01',
        sessions=[session] if session else None,
    )
    assert len(est) == estimations or total_estimations
    if session and test_auto:
        bold = layout.get(session=session, suffix='bold', extension='.nii.gz')[0]
        intended_rel = re.sub(r'^sub-[a-zA-Z0-9]*/', '', str(Path(bold).relative_to(layout.root)))
        b0_id = get_identifier(intended_rel)
        assert b0_id == ('auto_00000',)

    clear_registry()


def test_wrangler_medic_no_intent_does_not_fire(tmp_path):
    """Structure alone must not trigger MEDIC.

    A complex multi-echo BOLD with no ``B0FieldIdentifier`` and no
    ``IntendedFor`` carries no BIDS intent, so MEDIC must not be discovered
    (``fmapless=False`` rules out the ANAT fallback, isolating the MEDIC path).
    """
    bids_dir = str(tmp_path / 'medic_no_intent')
    generate_bids_skeleton(bids_dir, medic_no_intent)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', fmapless=False)
    assert est == []
    clear_registry()


@pytest.mark.parametrize(
    ('skeleton', 'no_medic', 'expected'),
    [
        # A: BIDS-recommended route — self-referential B0FieldIdentifier.
        (medic_b0_identifier, False, 3),
        # B: legacy route — IntendedFor on the complex BOLD sidecars.
        (medic, False, 3),
        # C/D: ``no_medic`` suppresses discovery via either route.
        (medic_b0_identifier, True, 0),
        (medic, True, 0),
    ],
    ids=['b0-identifier', 'intended-for', 'no_medic-b0', 'no_medic-intended-for'],
)
def test_wrangler_medic_trigger(tmp_path, skeleton, no_medic, expected):
    """Metadata-driven MEDIC discovery and the ``no_medic`` override.

    * **b0-identifier**: complex BOLD carries a self-referential
      ``B0FieldIdentifier`` (the BIDS-recommended route); discovered via Step 1.
    * **intended-for**: complex BOLD carries ``IntendedFor`` (legacy route);
      discovered via the dedicated MEDIC block.
    * **no_medic-***: ``no_medic=True`` skips MEDIC via either route, so
      nothing fires (``fmapless=False`` rules out the ANAT fallback).
    """
    bids_dir = str(tmp_path / 'medic_trigger')
    generate_bids_skeleton(bids_dir, skeleton)
    layout = gen_layout(bids_dir)
    estimators = find_estimators(
        layout=layout,
        subject='01',
        fmapless=False,
        no_medic=no_medic,
    )
    assert len(estimators) == expected
    for estimator in estimators:
        assert estimator.method.name == 'MEDIC'
        # 3 echoes × {mag, phase} per session.
        assert len(estimator.sources) == 6
    clear_registry()


def test_wrangler_medic_ordered_first(tmp_path):
    """MEDIC precedes static estimators in the returned list.

    Consumers (fMRIPrep) walk this list and select the first applicable
    estimator per target, so a dynamic MEDIC estimator must come before a
    coexisting static fieldmap. Here a PEPOLAR pair and a complex multi-echo
    BOLD each carry their own ``B0FieldIdentifier``; both are discovered and
    MEDIC must sort first regardless of ``B0FieldIdentifier`` iteration order.
    """
    skeleton = {
        '01': [
            {
                'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}],
                'func': [
                    {
                        'task': 'rest',
                        'run': run,
                        'suffix': 'bold',
                        'metadata': {
                            'RepetitionTime': 0.8,
                            'TotalReadoutTime': 0.5,
                            'PhaseEncodingDirection': ped,
                            'B0FieldIdentifier': 'pepolar1',
                            'B0FieldSource': 'pepolar1',
                        },
                    }
                    for run, ped in ((1, 'j'), (2, 'j-'))
                ]
                + [
                    {
                        'task': 'medic',
                        'echo': echo,
                        'part': part,
                        'suffix': 'bold',
                        'metadata': {
                            'EchoTime': te,
                            'RepetitionTime': 0.8,
                            'TotalReadoutTime': 0.5,
                            'PhaseEncodingDirection': 'j',
                            'B0FieldIdentifier': 'medic1',
                            **({'B0FieldSource': 'medic1'} if part == 'mag' else {}),
                        },
                    }
                    for echo, te in (('1', 0.0142), ('2', 0.0389))
                    for part in ('mag', 'phase')
                ],
            }
        ]
    }
    bids_dir = str(tmp_path / 'medic_first')
    generate_bids_skeleton(bids_dir, skeleton)
    layout = gen_layout(bids_dir)
    estimators = find_estimators(layout=layout, subject='01', fmapless=False)
    methods = [e.method.name for e in estimators]
    assert methods[0] == 'MEDIC', methods
    assert 'PEPOLAR' in methods
    clear_registry()


def test_single_reverse_pedir(tmp_path):
    bids_dir = tmp_path / 'bids'
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
    bids_dir = tmp_path / 'bids'

    T1w = {'suffix': 'T1w'}
    bold = {
        'task': 'rest',
        'suffix': 'bold',
        'metadata': {
            'RepetitionTime': 0.8,
            'TotalReadoutTime': 0.5,
            'PhaseEncodingDirection': 'j',
        },
    }
    me_metadata = [{'EchoTime': 0.01 * i, **bold['metadata']} for i in range(1, 4)]
    sbref = {**bold, **{'suffix': 'sbref'}}
    spec = {
        '01': {
            'anat': [T1w],
            'func': [bold],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', fmapless=True)
    assert len(est) == 1
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Multi-run generates one estimator per-run
    spec = {
        '01': {
            'anat': [T1w],
            'func': [{'run': i, **bold} for i in range(1, 3)],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', fmapless=True)
    assert len(est) == 2
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Multi-echo should only generate one estimator
    spec = {
        '01': {
            'anat': [T1w],
            'func': [{'echo': i + 1, **bold, **{'metadata': me_metadata[i]}} for i in range(3)],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', fmapless=True)
    assert len(est) == 1
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Matching bold+sbref should generate only one estimator
    spec = {
        '01': {
            'anat': [T1w],
            'func': [bold, sbref],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', fmapless=True)
    assert len(est) == 1
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Mismatching bold+sbref should generate two sbrefs
    spec = {
        '01': {
            'anat': [T1w],
            'func': [{'acq': 'A', **bold}, {'acq': 'B', **sbref}],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', fmapless=True)
    assert len(est) == 2
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)

    # Multiecho bold+sbref should generate only one estimator
    spec = {
        '01': {
            'anat': [T1w],
            'func': [{'echo': i + 1, **bold, **{'metadata': me_metadata[i]}} for i in range(3)]
            + [{'echo': i + 1, **sbref, **{'metadata': me_metadata[i]}} for i in range(3)],
        },
    }
    generate_bids_skeleton(bids_dir, spec)
    layout = gen_layout(bids_dir)
    est = find_estimators(layout=layout, subject='01', fmapless=True)
    assert len(est) == 1
    assert len(est[0].sources) == 2
    clear_registry()
    rmtree(bids_dir)
