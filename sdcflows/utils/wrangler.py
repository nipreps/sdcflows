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
"""Find fieldmaps on the BIDS inputs for :abbr:`SDC (susceptibility distortion correction)`."""
from itertools import product
from contextlib import suppress
from pathlib import Path


def find_estimators(*, layout, subject, fmapless=True, force_fmapless=False):
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
    Our ``ds000054`` dataset, created for *fMRIPrep*, only has one *phasediff* type of fieldmap
    with ``magnitude1`` and ``magnitude2`` files:

    >>> find_estimators(
    ...     layout=layouts['ds000054'],
    ...     subject="100185",
    ...     fmapless=False,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00000')]

    OpenNeuro's dataset with four *PEPOLAR* EPI files, two runs per phase-encoding direction
    (AP, PA):

    >>> find_estimators(
    ...     layout=layouts['ds001771'],
    ...     subject="36",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00001')]

    OpenNeuro's ``ds001600`` is an SDC test-dataset containing many different possibilities
    for fieldmap estimation:

    >>> find_estimators(
    ...     layout=layouts['ds001600'],
    ...     subject="1",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00002'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00003'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00004'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00005')]

    We can also pick one (simplified) HCP subject for testing purposes:

    >>> find_estimators(
    ...     layout=layouts['HCP101006'],
    ...     subject="101006",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00006'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00007')]

    Finally, *SDCFlows*' "*dataset A*" and "*dataset B*" contain BIDS structures
    with zero-byte NIfTI files and some corresponding metadata:

    >>> find_estimators(
    ...     layout=layouts['dsA'],
    ...     subject="01",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_...')]

    >>> find_estimators(
    ...     layout=layouts['dsB'],
    ...     subject="01",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_...')]

    After cleaning the registry, we can see how the "*fieldmap-less*" estimation
    can be forced:

    >>> from .. import fieldmaps as fm
    >>> fm.clear_registry()
    >>> find_estimators(
    ...     layout=layouts['ds000054'],
    ...     subject="100185",
    ...     fmapless={"bold"},
    ...     force_fmapless=True,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_00000'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.ANAT: 5>,
                        bids_id='auto_00001')]

    Likewise in a more comprehensive dataset:

    >>> find_estimators(
    ...     layout=layouts['ds001771'],
    ...     subject="36",
    ...     force_fmapless=True,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PEPOLAR: 2>,
                       bids_id='auto_00002'),
    FieldmapEstimation(sources=<7 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_00003'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_00004'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_00005')]

    Because "*dataset A*" contains very few metadata fields available, "*fieldmap-less*"
    heuristics come back empty (BOLD and DWI files are missing
    the mandatory ``PhaseEncodingDirection``, in this case):

    >>> find_estimators(
    ...     layout=layouts['dsA'],
    ...     subject="01",
    ...     force_fmapless=True,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.MAPPED: 4>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<3 files>, method=<EstimatorType.PHASEDIFF: 3>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_...')]

    This function should also correctly investigate multi-session datasets:

    >>> find_estimators(
    ...     layout=layouts['ds000206'],
    ...     subject="05",
    ...     fmapless=False,
    ...     force_fmapless=False,
    ... )  # doctest: +ELLIPSIS
    []

    >>> find_estimators(
    ...     layout=layouts['ds000206'],
    ...     subject="05",
    ...     fmapless=True,
    ...     force_fmapless=False,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                        bids_id='auto_00011'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_00012')]

    """
    from .. import fieldmaps as fm
    from bids.layout import Query

    base_entities = {
        "subject": subject,
        "extension": [".nii", ".nii.gz"],
        "scope": "raw",  # Ensure derivatives are not captured
    }

    subject_root = Path(layout.root) / f"sub-{subject}"

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
    sessions = layout.get_sessions()
    acqs = tuple(layout.get_acquisitions(suffix="epi") + [None])
    contrasts = tuple(layout.get_ceagents(suffix="epi") + [None])

    for ses, acq, ce in product(sessions or (None,), acqs, contrasts):
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
    has_intended = tuple()
    with suppress(ValueError):
        has_intended = layout.get(suffix="epi", IntendedFor=Query.ANY, **base_entities)

    for epi_fmap in has_intended:
        if epi_fmap.path in fm._estimators.sources:
            continue  # skip EPI images already considered above

        targets = [epi_fmap] + [
            layout.get_file(str(subject_root / intent))
            for intent in epi_fmap.get_metadata()["IntendedFor"]
        ]

        epi_sources = []
        for fmap in targets:
            with suppress(fm.MetadataError):
                epi_sources.append(
                    fm.FieldmapFile(fmap.path, metadata=fmap.get_metadata())
                )

        with suppress(ValueError, TypeError):
            estimators.append(fm.FieldmapEstimation(epi_sources))

    if estimators and not force_fmapless:
        fmapless = False

    # Find fieldmap-less schemes
    anat_file = layout.get(suffix="T1w", **base_entities)

    if not fmapless or not anat_file:
        return estimators

    from .epimanip import get_trt

    for ses, suffix in sorted(product(sessions or (None,), fmapless)):
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
            epi_targets.append(fm.FieldmapFile(candidate.path, metadata=meta))

        for pe_dir in sorted(set(pe_dirs)):
            pe_ro = [ro for ro, pe in zip(ro_totals, pe_dirs) if pe == pe_dir]
            for ro_time in sorted(set(pe_ro)):
                fmfiles, fmpaths = tuple(
                    zip(
                        *[
                            (target, str(Path(target.path).relative_to(subject_root)))
                            for i, target in enumerate(epi_targets)
                            if pe_dirs[i] == pe_dir and ro_totals[i] == ro_time
                        ]
                    )
                )
                estimators.append(
                    fm.FieldmapEstimation(
                        [
                            fm.FieldmapFile(
                                anat_file[0], metadata={"IntendedFor": fmpaths}
                            ),
                            *fmfiles,
                        ]
                    )
                )

    return estimators
