# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Estimate fieldmaps for :abbr:`SDC (susceptibility distortion correction)`."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import logging

from niworkflows.engine.workflows import LiterateWorkflow as Workflow


LOGGER = logging.getLogger('nipype.workflow')
DEFAULT_MEMORY_MIN_GB = 0.01


def init_fmap_preproc_wf(
    *,
    layout,
    omp_nthreads,
    output_dir,
    subject,
    debug=False,
    name='fmap_preproc_wf',
):
    """
    Stage the fieldmap data preprocessing steps of *SDCFlows*.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        An initialized PyBIDS layout.
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    output_dir : :obj:`str`
        Directory in which to save derivatives
    subject : :obj:`str`
        Participant label for this single-subject workflow.
    debug : :obj:`bool`
        Enable debugging outputs
    name : :obj:`str`, optional
        Workflow name (default: ``"fmap_preproc_wf"``)

    Examples
    --------
    >>> init_fmap_preproc_wf(
    ...     layout=layouts['ds001600'],
    ...     omp_nthreads=1,
    ...     output_dir="/tmp",
    ...     subject="1",
    ... )
    [FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>)]

    >>> init_fmap_preproc_wf(
    ...     layout=layouts['testdata'],
    ...     omp_nthreads=1,
    ...     output_dir="/tmp",
    ...     subject="HCP101006",
    ... )
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PHASEDIFF: 3>),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>)]

    """
    from ..fieldmaps import FieldmapEstimation, FieldmapFile

    base_entities = {
        "subject": subject,
        "extension": [".nii", ".nii.gz"],
        "space": None,  # Ensure derivatives are not captured
    }

    estimators = []

    # Set up B0 fieldmap strategies:
    for fmap in layout.get(
        suffix=["fieldmap", "phasediff", "phase1"], **base_entities
    ):
        e = FieldmapEstimation(
            FieldmapFile(fmap.path, metadata=fmap.get_metadata())
        )
        estimators.append(e)

    # A bunch of heuristics to select EPI fieldmaps
    sessions = layout.get_sessions() or [None]
    for session in sessions:
        dirs = layout.get_directions(
            suffix="epi",
            session=session,
            **base_entities,
        )
        if len(dirs) > 1:
            e = FieldmapEstimation([
                FieldmapFile(fmap.path, metadata=fmap.get_metadata())
                for fmap in layout.get(suffix="epi", session=session,
                                       direction=dirs, **base_entities)
            ])
            estimators.append(e)

    for e in estimators:
        LOGGER.info(
            f"{e.method}:: <{':'.join(s.path.name for s in e.sources)}>."
        )

    return estimators
