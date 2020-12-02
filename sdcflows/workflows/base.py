# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Estimate fieldmaps for :abbr:`SDC (susceptibility distortion correction)`."""
from itertools import product
from pathlib import Path
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
    ...     layout=layouts['ds000054'],
    ...     omp_nthreads=1,
    ...     output_dir="/tmp",
    ...     subject="100185",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...')]

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
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...')]

    """
    from .. import fieldmaps as fm
    from bids.layout import Query

    base_entities = {
        "subject": subject,
        "extension": [".nii", ".nii.gz"],
        "space": None,  # Ensure derivatives are not captured
    }

    estimators = []

    # Set up B0 fieldmap strategies:
    for fmap in layout.get(suffix=["fieldmap", "phasediff", "phase1"], **base_entities):
        e = fm.FieldmapEstimation(
            fm.FieldmapFile(fmap.path, metadata=fmap.get_metadata())
        )
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
            e = fm.FieldmapEstimation(
                [
                    fm.FieldmapFile(fmap.path, metadata=fmap.get_metadata())
                    for fmap in layout.get(direction=dirs, **entities)
                ]
            )
            estimators.append(e)

    # At this point, only single-PE _epi files WITH ``IntendedFor`` can be automatically processed
    # (this will be easier with bids-standard/bids-specification#622 in).
    try:
        has_intended = layout.get(suffix="epi", IntendedFor=Query.ANY, **base_entities)
    except ValueError:
        has_intended = tuple()

    for epi_fmap in has_intended:
        if epi_fmap.path in fm._estimators.sources:
            continue  # skip EPI images already considered above

        subject_root = Path(epi_fmap.path.rpartition("/sub-")[0]).parent
        targets = [epi_fmap] + [
            layout.get_file(str(subject_root / intent))
            for intent in epi_fmap.get_metadata()["IntendedFor"]
        ]

        epi_sources = []
        for fmap in targets:
            try:
                epi_sources.append(
                    fm.FieldmapFile(fmap.path, metadata=fmap.get_metadata())
                )
            except fm.MetadataError:
                pass

        try:
            estimators.append(fm.FieldmapEstimation(epi_sources))
        except (ValueError, TypeError) as exc:
            LOGGER.warning(
                f"FieldmapEstimation strategy failed for <{epi_fmap.path}>. Reason: {exc}."
            )

    for e in estimators:
        LOGGER.info(f"{e.method}:: <{':'.join(s.path.name for s in e.sources)}>.")

    return estimators
