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
from __future__ import annotations
import logging
from functools import reduce
from itertools import product
from contextlib import suppress
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from bids.layout import BIDSLayout, BIDSFile
from bids.utils import listify

from .. import fieldmaps as fm


def _normalize_intent(
    intent: str,
    subject: str
) -> str | None:
    """Convert BIDS-URI intent to subject-relative intent

    SDCFlows currently makes strong assumptions about old-style intents,
    and a change to that needs to be carefully considered and tested.
    """
    if intent.startswith("bids::"):
        # bids::sub-<subject>/
        #          ^- 10     ^- 11
        return intent[11 + len(subject):]
    return intent


def _resolve_intent(
    intent: str,
    layout: BIDSLayout,
    subject: str
) -> str | None:
    root = Path(layout.root)
    if intent.startswith("bids::"):
        return str(root / intent[6:])
    if not intent.startswith("bids:"):
        return str(root / f"sub-{subject}" / intent)
    return intent


def _filter_metadata(
    metadata: Dict[str, Any],
    subject: str
) -> Dict[str, Any]:
    intents = metadata.get("IntendedFor")
    if intents:
        updated = [_normalize_intent(intent, subject) for intent in listify(intents)]
        return {**metadata, "IntendedFor": updated}
    return metadata


def find_estimators(
    *,
    layout: BIDSLayout,
    subject: str,
    sessions: Optional[List[str]] = None,
    fmapless: Union[bool, set] = True,
    force_fmapless: bool = False,
    logger: Optional[logging.Logger] = None,
    bids_filters: Optional[dict] = None,
    anat_suffix: Union[str, List[str]] = 'T1w',
) -> list:
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
    sessions : :obj:`list` or None
        One of more session identifiers. If None, all sessions will be used.
    fmapless : :obj:`bool` or :obj:`set`
        Indicates if fieldmap-less heuristics should be executed.
        When ``fmapless`` is a :obj:`set`, it can contain valid BIDS suffixes
        for EPI images (namely, ``"dwi"``, ``"bold"``, ``"asl"``, or ``"sbref"``).
        When ``fmapless`` is ``True``, heuristics will use the ``{"bold", "dwi", "asl"}`` set.
    force_fmapless : :obj:`bool`
        When some other fieldmap estimation methods have been found, fieldmap-less
        estimation will be skipped except if ``force_fmapless`` is ``True``.
    logger
        The logger used to relay messages. If not provided, one will be created.
    bids_filters
        Optional dictionary of key/values to filter the entities on.
        This allows lower level file inclusion/exclusion.
    anat_suffix : :obj:`str` or :obj:`list`
        String or list of strings to filter anatomical images for fieldmap-less
        approaches. If not provided, ``T1w`` is used.

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
                        bids_id='auto_00007'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='auto_00008')]

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
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                        bids_id='auto_...'),
     FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                        bids_id='auto_...')]

    Likewise in a more comprehensive dataset:

    >>> find_estimators(
    ...     layout=layouts['ds001771'],
    ...     subject="36",
    ...     force_fmapless=True,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<4 files>, method=<EstimatorType.PEPOLAR: 2>,
                       bids_id='auto_...'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...')]

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
                        bids_id='auto_...'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...')]

    When the ``B0FieldIdentifier`` metadata is set for one or more fieldmaps, then
    the heuristics that use ``IntendedFor`` are dismissed:

    >>> find_estimators(
    ...     layout=layouts['dsC'],
    ...     subject="01",
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<5 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='pepolar4pe')]

    The only exception to the priority of ``B0FieldIdentifier`` is when fieldmaps
    are searched with the ``force_fmapless`` argument on:

    >>> fm.clear_registry()  # Necessary as `pepolar4pe` is not changing.
    >>> find_estimators(
    ...     layout=layouts['dsC'],
    ...     subject="01",
    ...     fmapless=True,
    ...     force_fmapless=True,
    ... )  # doctest: +ELLIPSIS
    [FieldmapEstimation(sources=<5 files>, method=<EstimatorType.PEPOLAR: 2>,
                        bids_id='pepolar4pe'),
    FieldmapEstimation(sources=<2 files>, method=<EstimatorType.ANAT: 5>,
                       bids_id='auto_...')]

    """
    from .misc import create_logger
    from bids.layout import Query
    from bids.exceptions import BIDSEntityError

    # The created logger is set to ERROR log level
    logger = logger or create_logger('sdcflows.wrangler')

    base_entities = {
        "subject": subject,
        "extension": [".nii", ".nii.gz"],
        "part": ["mag", None],
        "scope": "raw",  # Ensure derivatives are not captured
    }

    if bids_filters:
        filters = bids_filters.copy()  # copy to avoid altering in place
        if 'session' in bids_filters and sessions is not None:
            raise ValueError("Filters include session, but session is already defined.")
        sessions = listify(filters.pop('session', None))
        base_entities.update(filters)

    subject_root = Path(layout.root) / f"sub-{subject}"
    sessions = sessions or layout.get_sessions(subject=subject) or [None]
    fmapless = fmapless or {}
    if fmapless is True:
        fmapless = {"bold", "dwi", "asl"}

    estimators = []

    # Step 1. Use B0FieldIdentifier metadata
    b0_ids = tuple()
    with suppress(BIDSEntityError):
        # flatten lists from json (tupled in pybids for hashing), then unique
        b0_ids = reduce(
            set.union,
            (listify(ids) for ids in layout.get_B0FieldIdentifiers(**base_entities)),
            set()
        )

    if b0_ids:
        logger.debug(
            "Dataset includes `B0FieldIdentifier` metadata."
            "Any data missing this metadata will be ignored."
        )

        for b0_id in b0_ids:
            # Found B0FieldIdentifier metadata entries
            b0_entities = base_entities.copy()
            b0_entities["B0FieldIdentifier"] = b0_id

            bare_ids = layout.get(**base_entities, B0FieldIdentifier=b0_id)
            listed_ids = layout.get(
                **base_entities,
                B0FieldIdentifier=f'"{b0_id}"',  # Double quotes to match JSON, not Python repr
                regex_search=True,
            )
            try:
                e = fm.FieldmapEstimation(
                    [
                        fm.FieldmapFile(fmap.path, metadata=fmap.get_metadata())
                        for fmap in bare_ids + listed_ids
                    ]
                )
            except (ValueError, TypeError) as err:
                _log_debug_estimator_fail(
                    logger, b0_id, bare_ids + listed_ids, layout.root, str(err)
                )
            else:
                _log_debug_estimation(logger, e, layout.root)
                estimators.append(e)

    # Step 2. If no B0FieldIdentifiers were found, try several heuristics
    if not estimators:
        # Set up B0 fieldmap strategies:
        for fmap in layout.get(
            **{
                **base_entities,
                **{'suffix': ["fieldmap", "phasediff", "phase1"], 'session': sessions}
            }
        ):
            try:
                e = fm.FieldmapEstimation(
                    fm.FieldmapFile(
                        fmap.path,
                        metadata=_filter_metadata(fmap.get_metadata(), subject),
                    )
                )
            except (ValueError, TypeError) as err:
                _log_debug_estimator_fail(
                    logger, "unnamed fieldmap", [fmap], layout.root, str(err)
                )
            else:
                _log_debug_estimation(logger, e, layout.root)
                estimators.append(e)

        # A bunch of heuristics to select EPI fieldmaps
        acqs = (
            base_entities.get('acquisitions')
            or layout.get_acquisitions(subject=subject, suffix="epi") + [None]
        )
        contrasts = (
            base_entities.get('ceagent')
            or layout.get_ceagents(subject=subject, suffix="epi") + [None]
        )
        for ses, acq, ce in product(sessions, acqs, contrasts):
            entities = base_entities.copy()
            entities.update(
                {"suffix": "epi", "session": ses, "acquisition": acq, "ceagent": ce}
            )
            dirs = layout.get_directions(**entities)
            if len(dirs) > 1:
                by_intent = {}
                for fmap in layout.get(**{**entities, **{'direction': dirs}}):
                    fmapfile = fm.FieldmapFile(
                        fmap.path,
                        metadata=_filter_metadata(fmap.get_metadata(), subject),
                    )
                    by_intent.setdefault(
                        tuple(fmapfile.metadata.get('IntendedFor', ())), []
                    ).append(fmapfile)
                for collection in by_intent.values():
                    try:
                        e = fm.FieldmapEstimation(collection)
                    except (ValueError, TypeError) as err:
                        _log_debug_estimator_fail(
                            logger, "unnamed PEPOLAR", collection, layout.root, str(err)
                        )
                    else:
                        _log_debug_estimation(logger, e, layout.root)
                        estimators.append(e)

        # At this point, only single-PE _epi files WITH ``IntendedFor`` can
        # be automatically processed.
        has_intended = tuple()
        with suppress(ValueError):
            has_intended = layout.get(
                **{
                    **base_entities,
                    **{'suffix': 'epi', 'IntendedFor': Query.REQUIRED, 'session': sessions}
                }
            )

        for epi_fmap in has_intended:
            if epi_fmap.path in fm._estimators.sources:
                logger.debug("Skipping fieldmap %s (already in use)", epi_fmap.relpath)
                continue  # skip EPI images already considered above

            logger.debug("Found single PE fieldmap %s", epi_fmap.relpath)
            epi_base_md = epi_fmap.get_metadata()

            # Find existing IntendedFor targets and warn if missing
            all_targets = []
            for intent in listify(epi_base_md["IntendedFor"]):
                target = layout.get_file(_resolve_intent(intent, layout, subject))
                if target is None:
                    logger.debug("Single PE target %s not found", intent)
                    continue
                all_targets.append(target)

            # If sbrefs are targets, then the goal is generally to estimate with epi+sbref
            # and correct bold/dwi/asl
            sbrefs = [
                target for target in all_targets if target.entities["suffix"] == "sbref"
            ]
            if sbrefs:
                targets = sbrefs
                intent_map = []
                for sbref in sbrefs:
                    ents = sbref.get_entities(metadata=False)
                    ents["suffix"] = ["bold", "dwi", "asl"]
                    intent_map.append(
                        [
                            target
                            for target in layout.get(**ents)
                            if target in all_targets
                        ]
                    )
            else:
                targets = all_targets
                intent_map = [[target] for target in all_targets]

            for target, intent in zip(targets, intent_map):
                logger.debug("Found single PE target %s", target.relpath)
                # The new estimator is IntendedFor the individual targets,
                # even if the EPI file is IntendedFor multiple
                estimator_md = epi_base_md.copy()
                estimator_md["IntendedFor"] = [
                    str(Path(pathlike).relative_to(subject_root))
                    for pathlike in intent
                ]
                try:
                    e = fm.FieldmapEstimation(
                        [
                            fm.FieldmapFile(epi_fmap.path, metadata=estimator_md),
                            fm.FieldmapFile(target.path, metadata=target.get_metadata())
                        ]
                    )
                except (ValueError, TypeError) as err:
                    _log_debug_estimator_fail(
                        logger,
                        "unnamed PEPOLAR",
                        [epi_fmap, target],
                        layout.root,
                        str(err)
                    )
                else:
                    _log_debug_estimation(logger, e, layout.root)
                    estimators.append(e)

    if estimators and not force_fmapless:
        fmapless = False

    # Find fieldmap-less schemes
    anat_file = layout.get(**{**base_entities, **{'suffix': anat_suffix, 'session': sessions}})

    if not fmapless or not anat_file:
        logger.debug("Skipping fmap-less estimation")
        return estimators

    logger.debug("Attempting fmap-less estimation")
    estimator_specs = find_anatomical_estimators(
        anat_file=anat_file[0],
        layout=layout,
        subject=subject,
        sessions=sessions,
        base_entities=base_entities,
        suffixes=fmapless,
    )
    for spec in estimator_specs:
        try:
            estimator = fm.FieldmapEstimation(spec)
        except (ValueError, TypeError) as err:
            _log_debug_estimator_fail(logger, "ANAT", spec, layout.root, str(err))
        else:
            _log_debug_estimation(logger, estimator, layout.root)
            estimators.append(estimator)
    return estimators


def find_anatomical_estimators(
    *,
    anat_file: BIDSFile,
    layout: BIDSLayout,
    subject: str,
    sessions: List[str],
    base_entities: Dict[str, Any],
    suffixes: List[str],
) -> List[List[fm.FieldmapFile]]:
    r"""Find anatomical estimators

    Given an anatomical reference image, create lists of files for estimating
    susceptibility distortion for the EPI images in a dataset.

    Parameters
    ----------
    anat_file : :class:`bids.layout.BIDSFile`
        Anatomical reference image to use in estimators.
    layout : :class:`bids.layout.BIDSLayout`
        An initialized PyBIDS layout.
    subject : :class:`str`
        Participant label for this single-subject workflow.
    sessions : :class:`list`
        One of more session identifiers. To use all, pass ``[None]``.
    base_entities : :class:`dict`
        Entities to use to query for images. These should include any filters.
    suffixes : :class:`list`
        EPI suffixes, for example ``["bold", "dwi", "asl"]``. Associated ``"sbref"``\s
        will be found and used in place of BOLD/diffusion EPIs.
        Similarly, ``"m0scan"``\s associated with ASL runs with the ``IntendedFor`` or
        ``B0FieldIdentifier`` metadata will be used in place of ASL runs.
    """

    from .epimanip import get_trt

    subject_root = Path(layout.root) / f"sub-{subject}"

    hits = set()  # Avoid duplicates
    estimators = []
    for ses, suffix in sorted(product(sessions, suffixes)):
        suffixes = ["sbref", suffix]  # Order indicates preference; prefer sbref
        datatype = {
            "bold": "func",
            "dwi": "dwi",
            "asl": "perf",
        }[suffix]
        candidates = layout.get(
            **{
                **base_entities,
                **{"suffix": suffixes, "session": ses, "datatype": datatype},
            }
        )

        # Filter out candidates without defined PE direction
        epi_targets = []

        for candidate in candidates:
            meta = candidate.get_metadata()

            if not meta.get("PhaseEncodingDirection"):
                continue

            with suppress(ValueError):
                meta.update({"TotalReadoutTime": get_trt(meta, candidate.path)})
            epi_targets.append(fm.FieldmapFile(candidate, metadata=meta))

        def sort_key(fmap):
            # Return sbref before DWI/BOLD and shortest echo first
            return suffixes.index(fmap.suffix), fmap.metadata.get("EchoTime", 1)

        for target in sorted(epi_targets, key=sort_key):
            if target.path in hits:
                continue
            query = {**base_entities, **target.entities}

            # Find all echos, so strip from query, if present
            query.pop("echo", None)

            # Include sbref and EPI images in IntendedFor
            # No harm in including sbrefs that won't be corrected,
            # and ensures the hits set prevents doubling up
            intent = [Path(epi) for epi in layout.get(suffix=suffixes, **query)]
            metadata = {
                "IntendedFor": [str(epi.relative_to(subject_root)) for epi in intent]
            }
            estimators.append([fm.FieldmapFile(anat_file, metadata=metadata), target])
            hits.update(intent)

    return estimators


def _log_debug_estimation(
    logger: logging.Logger,
    estimation: fm.FieldmapEstimation,
    bids_root: str,
) -> None:
    """A helper function to log estimation information when running with verbosity."""
    logger.debug(
        "Found %s estimation from %d sources:\n- %s",
        estimation.method.name,
        len(estimation.sources),
        "\n- ".join(
            [str(Path(s.path).relative_to(bids_root)) for s in estimation.sources]
        ),
    )


def _log_debug_estimator_fail(
    logger: logging.Logger,
    b0_id: str,
    files: List[BIDSFile],
    bids_root: str,
    message: str
) -> None:
    """A helper function to log failures to build an estimator when running with verbosity."""
    logger.debug(
        "Failed to construct %s estimation from %d sources:\n- %s\nError: %s",
        b0_id,
        len(files),
        "\n- ".join([str(Path(s.path).relative_to(bids_root)) for s in files]),
        message,
    )
