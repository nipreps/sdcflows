# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Estimate fieldmaps for :abbr:`SDC (susceptibility distortion correction)`."""
from itertools import product
from nipype import logging

LOGGER = logging.getLogger("nipype.workflow")
DEFAULT_MEMORY_MIN_GB = 0.01


def init_fmap_preproc_wf(
    *, layout, omp_nthreads, output_dir, subject, debug=False, name="fmap_preproc_wf",
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
    ...     layout=layouts['ds001771'],
    ...     omp_nthreads=1,
    ...     output_dir="/tmp",
    ...     subject="36",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...')]

    >>> init_fmap_preproc_wf(
    ...     layout=layouts['ds001600'],
    ...     omp_nthreads=1,
    ...     output_dir="/tmp",
    ...     subject="1",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...')]

    >>> init_fmap_preproc_wf(
    ...     layout=layouts['HCP101006'],
    ...     omp_nthreads=1,
    ...     output_dir="/tmp",
    ...     subject="101006",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...')]

    >>> init_fmap_preproc_wf(
    ...     layout=layouts['dsA'],
    ...     omp_nthreads=1,
    ...     output_dir="/tmp",
    ...     subject="01",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>, bids_id='...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...')]

    """
    from ..fieldmaps import FieldmapEstimation, FieldmapFile

    base_entities = {
        "subject": subject,
        "extension": [".nii", ".nii.gz"],
        "space": None,  # Ensure derivatives are not captured
    }

    estimators = []

    # Set up B0 fieldmap strategies:
    for fmap in layout.get(suffix=["fieldmap", "phasediff", "phase1"], **base_entities):
        e = FieldmapEstimation(FieldmapFile(fmap.path, metadata=fmap.get_metadata()))
        estimators.append(e)

    # A bunch of heuristics to select EPI fieldmaps
    sessions = layout.get_sessions() or (None,)
    acqs = tuple(layout.get_acquisitions(suffix="epi") + [None])
    contrasts = tuple(layout.get_ceagents(suffix="epi") + [None])

    for ses, acq, ce in product(sessions, acqs, contrasts):
        entities = base_entities.copy()
        entities.update(
            {"suffix": "epi", "session": ses, "acquisition": acq, "ceagent": ce}
        )
        dirs = layout.get_directions(**entities)
        if len(dirs) > 1:
            e = FieldmapEstimation(
                [
                    FieldmapFile(fmap.path, metadata=fmap.get_metadata())
                    for fmap in layout.get(direction=dirs, **entities)
                ]
            )
            estimators.append(e)

    for e in estimators:
        LOGGER.info(f"{e.method}:: <{':'.join(s.path.name for s in e.sources)}>.")

    return estimators
