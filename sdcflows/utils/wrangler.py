# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Find fieldmaps on the BIDS inputs for :abbr:`SDC (susceptibility distortion correction)`."""
from itertools import product
from contextlib import suppress
from pathlib import Path


def find_estimators(layout, subject=None, fmapless=True, force_fmapless=False):
    """
    Apply basic heuristics to automatically find available data for fieldmap estimation.

    The "*fieldmap-less*" heuristics only attempt to find ``_dwi`` and ``_bold`` candidates
    to pair with a ``_T1w`` anatomical reference.
    For more complicated heuristics (for instance, using ``_T2w`` images or ``_sbref``
    images,) the :py:class:`~sdcflows.fieldmaps.FieldmapEstimation` object must be
    created manually by the user.

    Parameters
    ----------
    layout : :obj:`bids.layout.BIDSLayout`
        An initialized PyBIDS layout.
    subject : :obj:`str`
        Participant label for this single-subject workflow.
    fmapless : :obj:`bool` or :obj:`set`
        Indicates if fieldmap-less heuristics should be executed.
        When ``fmapless`` is a :obj:`set`, it can contain valid BIDS suffices
        for EPI images (namely, ``"dwi"``, ``"bold"``, or ``"sbref"``).
        When ``fmapless`` is ``True``, heuristics will use the ``{"bold", "dwi"}`` set.
    force_fmapless : :obj:`bool`
        When some other fieldmap estimation methods have been found, fieldmap-less
        estimation will be skipped except if ``force_fmapless`` is ``True``.

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
    ...     fmapless=False,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00000')]

    >>> find_estimators(
    ...     layout=layouts['ds001771'],
    ...     subject="36",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>, bids_id='auto_00001'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>, bids_id='auto_00002'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00003'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00004')]

    >>> find_estimators(
    ...     layout=layouts['ds001600'],
    ...     subject="1",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00005'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00006'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00007'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00008')]

    >>> find_estimators(
    ...     layout=layouts['HCP101006'],
    ...     subject="101006",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00009'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00010')]

    >>> find_estimators(
    ...     layout=layouts['dsA'],
    ...     subject="01",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>,
                        bids_id='auto_00011'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00012'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00013'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00014'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00015')]

    >>> from .. import fieldmaps as fm
    >>> fm.clear_registry()
    >>> find_estimators(
    ...     layout=layouts['ds000054'],
    ...     subject="100185",
    ...     force_fmapless=True,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00000'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.ANAT: 5>,
                        bids_id='auto_00001')]

    >>> find_estimators(
    ...     layout=layouts['ds001771'],
    ...     subject="36",
    ...     force_fmapless=True,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>,
                       bids_id='auto_00002'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>,
                       bids_id='auto_00003'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                       bids_id='auto_00004'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                       bids_id='auto_00005'),
    FieldmapEstimation(sources=<7 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_00006'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_00007'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_00008')]

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

    fmapless = fmapless or {}
    if fmapless is True:
        fmapless = {"bold", "dwi"}

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

    if estimators and not force_fmapless:
        fmapless = False

    if not fmapless:
        return estimators

    # Find fieldmap-less schemes
    anat_file = layout.get(suffix="T1w", desc=None, **base_entities)

    if not anat_file:
        return estimators

    from .epimanip import get_trt
    intended_root = Path(anat_file[0].path).parent.parent
    for ses, suffix in sorted(product(sessions, fmapless)):
        candidates = layout.get(suffix=suffix, session=ses, **base_entities)

        # Filter out candidates without defined PE direction
        epi_targets = []
        pe_dirs = []
        ro_totals = []

        for candidate in candidates:
            meta = candidate.get_metadata()
            pe_dir = meta.get("PhaseEncodingDirection")
            if not pe_dir:
                continue

            pe_dirs.append(pe_dir)
            ro = 1.0
            with suppress(ValueError):
                ro = get_trt(meta, candidate.path)
            ro_totals.append(ro)
            meta.update({"TotalReadoutTime": ro})
            epi_targets.append(
                fm.FieldmapFile(candidate.path, metadata=meta)
            )

        for pe_dir in sorted(set(pe_dirs)):
            pe_ro = [ro for ro, pe in zip(ro_totals, pe_dirs) if pe == pe_dir]
            for ro_time in sorted(set(pe_ro)):
                fmfiles, fmpaths = tuple(zip(*[
                    (target, str(Path(target.path).relative_to(intended_root)))
                    for i, target in enumerate(epi_targets)
                    if pe_dirs[i] == pe_dir and ro_totals[i] == ro_time
                ]))
                estimators.append(
                    fm.FieldmapEstimation(
                        [fm.FieldmapFile(
                            anat_file[0],
                            metadata={"IntendedFor": fmpaths}
                        ), *fmfiles]
                    )
                )
                # import pdb; pdb.set_trace()

    return estimators
