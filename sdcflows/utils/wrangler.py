# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Find fieldmaps on the BIDS inputs for :abbr:`SDC (susceptibility distortion correction)`."""
from itertools import product
from pathlib import Path


def find_estimators(layout, subject=None):
    """
    Apply basic heuristics to automatically find available data for fieldmap estimation.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        An initialized PyBIDS layout.
    subject : :obj:`str`
        Participant label for this single-subject workflow.

    Returns
    -------
    estimators : :obj:`list`
        The list of :py:class:`~sdcflows.fieldmaps.FieldmapEstimation` objects that have
        successfully been built (meaning, all necessary inputs and corresponding metadata
        are present in the given layout.)

    Examples
    --------
    >>> find_estimators(
    ...     layout=layouts['ds000054'],
    ...     subject="100185",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...')]

    >>> find_estimators(
    ...     layout=layouts['ds001771'],
    ...     subject="36",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...')]

    >>> find_estimators(
    ...     layout=layouts['ds001600'],
    ...     subject="1",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...')]

    >>> find_estimators(
    ...     layout=layouts['HCP101006'],
    ...     subject="101006",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PHASEDIFF: 3>, bids_id='...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, bids_id='...')]

    >>> find_estimators(
    ...     layout=layouts['dsA'],
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

    if subject is None:
        subject = Query.ANY

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
        except (ValueError, TypeError):
            pass

    return estimators
