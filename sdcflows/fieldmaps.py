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
"""Utilities for fieldmap estimation."""
from pathlib import Path
from enum import Enum, auto
from collections import defaultdict
import re
import attr
from json import loads
from bids.layout import parse_file_entities
from bids.utils import listify
from niworkflows.utils.bids import relative_to_root
from .utils.bimap import EstimatorRegistry


_estimators = EstimatorRegistry()
_intents = defaultdict(set)


class MetadataError(ValueError):
    """A better name for a specific value error."""


class EstimatorType(Enum):
    """Represents different types of fieldmap estimation approach."""

    UNKNOWN = auto()
    PEPOLAR = auto()
    PHASEDIFF = auto()
    MAPPED = auto()
    ANAT = auto()


MODALITIES = {
    "bold": EstimatorType.PEPOLAR,
    "dwi": EstimatorType.PEPOLAR,
    "epi": EstimatorType.PEPOLAR,
    "fieldmap": EstimatorType.MAPPED,
    "magnitude": None,
    "magnitude1": None,
    "magnitude2": None,
    "phase1": EstimatorType.PHASEDIFF,
    "phase2": EstimatorType.PHASEDIFF,
    "phasediff": EstimatorType.PHASEDIFF,
    "sbref": EstimatorType.PEPOLAR,
    "T1w": EstimatorType.ANAT,
    "T2w": EstimatorType.ANAT,
}


def _type_setter(obj, attribute, value):
    """Make sure the type of estimation is not changed."""
    if obj.method == value:
        return value
    elif obj.method != EstimatorType.UNKNOWN:
        raise TypeError(f"Cannot change determined method {obj.method} to {value}.")

    if value not in (
        EstimatorType.PEPOLAR,
        EstimatorType.PHASEDIFF,
        EstimatorType.MAPPED,
        EstimatorType.ANAT,
    ):
        raise ValueError(f"Invalid estimation method type {value}.")

    return value


def _id_setter(obj, attribute, value):
    if obj.bids_id and obj.bids_id == value:
        return value
    raise ValueError("Cannot edit the bids_id")


@attr.s(slots=True)
class FieldmapFile:
    """
    Represent a file that can be used in some fieldmap estimation method.

    The class will read metadata from a sidecar JSON with filename matching that
    of the file.
    This class may receive metadata as a keyword argument at initialization.

    Examples
    --------
    >>> f = FieldmapFile(dsA_dir / "sub-01" / "anat" / "sub-01_T1w.nii.gz")
    >>> f.suffix
    'T1w'

    >>> FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_dir-LR_epi.nii.gz",
    ...     find_meta=False
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    MetadataError:

    >>> f = FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_dir-LR_epi.nii.gz",
    ... )
    >>> f.metadata['TotalReadoutTime']
    0.005

    >>> f = FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_dir-LR_epi.nii.gz",
    ...     metadata={"TotalReadoutTime": None},
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    MetadataError:

    >>> f = FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_dir-LR_epi.nii.gz",
    ...     metadata={'TotalReadoutTime': 0.006}
    ... )
    >>> f.metadata['TotalReadoutTime']
    0.006

    >>> FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_phasediff.nii.gz",
    ...     find_meta=False
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    MetadataError:

    >>> f = FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_phasediff.nii.gz"
    ... )
    >>> f.metadata['EchoTime2']
    0.00746

    >>> FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_phase2.nii.gz",
    ...     find_meta=False
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    MetadataError:

    >>> f = FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_phase2.nii.gz"
    ... )
    >>> f.metadata['EchoTime']
    0.00746

    >>> FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_fieldmap.nii.gz",
    ...     find_meta=False
    ... )  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    MetadataError:

    >>> f = FieldmapFile(
    ...     dsA_dir / "sub-01" / "fmap" / "sub-01_fieldmap.nii.gz"
    ... )
    >>> f.metadata['Units']
    'rad/s'

    """

    path = attr.ib(converter=Path, repr=str, on_setattr=attr.setters.NO_OP)
    """Path to a fieldmap file."""

    entities = attr.ib(init=False, repr=False)
    """BIDS entities extracted from filepath."""

    suffix = attr.ib(init=False, repr=False)
    """Extracted suffix from input file."""

    bids_root = attr.ib(init=False, default=None, repr=False)
    """Path of the BIDS root."""

    metadata = attr.ib(kw_only=True, default=attr.Factory(dict))
    """
    Metadata associated to this file. When provided as keyword argument in initialization,
    will overwrite metadata read from sidecar JSON.
    """

    find_meta = attr.ib(kw_only=True, default=True, type=bool, repr=False)
    """Enable/disable automated search for corresponding metadata."""

    @path.validator
    def check_path(self, attribute, value):
        """Validate a fieldmap path."""
        if not value.is_file():
            raise FileNotFoundError(
                f"File path <{value}> does not exist, is a broken link, or it is not a file"
            )

        if not value.name.endswith((".nii", ".nii.gz")):
            raise ValueError(f"File path <{value}> does not look like a NIfTI file.")

        suffix = re.search(r"(?<=_)\w+(?=\.nii)", value.name).group()
        if suffix not in tuple(MODALITIES.keys()):
            raise ValueError(
                f"File path <{value}> with suffix <{suffix}> is not a valid "
                "fieldmap sourcefile."
            )

    def __attrs_post_init__(self):
        """Validate metadata and additional checks."""
        self.entities = parse_file_entities(str(self.path))
        self.suffix = self.entities.pop("suffix")
        extension = self.entities.pop("extension").lstrip(".")

        # Automatically fill metadata in when possible
        # TODO: implement BIDS hierarchy of metadata
        if self.find_meta:
            sidecar = Path(str(self.path).replace(extension, "json"))
            if sidecar.is_file():
                _meta = self.metadata or {}
                self.metadata = loads(sidecar.read_text())
                self.metadata.update(_meta)

        # Attempt to infer a bids_root folder
        relative_path = relative_to_root(self.path)
        self.bids_root = (
            Path(str(self.path)[: -len(str(relative_path))])
            if str(relative_path) != str(self.path)
            else None
        )

        if self.suffix in ("T1w", "T2w"):
            self.metadata["TotalReadoutTime"] = 0.0

        # Check for REQUIRED metadata (depends on suffix.)
        if self.suffix in ("bold", "dwi", "epi", "sbref"):
            if "PhaseEncodingDirection" not in self.metadata:
                raise MetadataError(
                    f"Missing 'PhaseEncodingDirection' for <{self.path}>."
                )

            from .utils.epimanip import get_trt

            try:
                get_trt(self.metadata, in_file=self.path)
            except ValueError as exc:
                raise MetadataError(
                    f"Missing readout timing information for <{self.path}>."
                ) from exc

        elif self.suffix == "fieldmap" and "Units" not in self.metadata:
            raise MetadataError(f"Missing 'Units' for <{self.path}>.")

        elif self.suffix == "phasediff" and (
            "EchoTime1" not in self.metadata or "EchoTime2" not in self.metadata
        ):
            raise MetadataError(
                f"Missing 'EchoTime1' and/or 'EchoTime2' for <{self.path}>."
            )

        elif self.suffix in ("phase1", "phase2") and ("EchoTime" not in self.metadata):
            raise MetadataError(f"Missing 'EchoTime' for <{self.path}>.")


@attr.s(slots=True)
class FieldmapEstimation:
    """
    Represent fieldmap estimation strategies.

    This class provides a consistent interface to all types of fieldmap estimation
    strategies.
    The actual type of method for estimation is inferred from the ``sources`` input,
    and collects all the available metadata.

    """

    sources = attr.ib(
        default=None,
        converter=lambda v: [
            FieldmapFile(f) if not isinstance(f, FieldmapFile) else f
            for f in listify(v)
        ],
        repr=lambda v: f"<{len(v)} files>",
    )
    """File path or list of paths indicating the source data to estimate a fieldmap."""

    method = attr.ib(init=False, default=EstimatorType.UNKNOWN, on_setattr=_type_setter)
    """Flag indicating the estimator type inferred from the input sources."""

    bids_id = attr.ib(default=None, kw_only=True, type=str, on_setattr=_id_setter)
    """The unique ``B0FieldIdentifier`` field of this fieldmap."""

    _wf = attr.ib(init=False, default=None, repr=False)
    """Internal pointer to a workflow."""

    def __attrs_post_init__(self):
        """Determine the intended fieldmap estimation type and check for data completeness."""
        suffix_list = [f.suffix for f in self.sources]
        suffix_set = set(suffix_list)

        # Fieldmap option 1: actual field-mapping sequences
        fmap_types = suffix_set.intersection(
            ("fieldmap", "phasediff", "phase1", "phase2")
        )
        if len(fmap_types) > 1 and fmap_types - set(("phase1", "phase2")):
            raise TypeError(f"Incompatible suffices found: <{','.join(fmap_types)}>.")

        if fmap_types:
            sources = sorted(
                str(f.path)
                for f in self.sources
                if f.suffix in ("fieldmap", "phasediff", "phase1", "phase2")
            )

            # Automagically add the corresponding phase2 file if missing as argument
            missing_phases = ("phase1" not in fmap_types, "phase2" not in fmap_types)
            if sum(missing_phases) == 1:
                mis_ph = "phase1" if missing_phases[0] else "phase2"
                hit_ph = "phase2" if missing_phases[0] else "phase1"
                new_source = sources[0].replace(hit_ph, mis_ph)
                self.sources.append(FieldmapFile(new_source))
                sources.insert(int(missing_phases[1]), new_source)

            # Set method, this cannot be undone
            self.method = MODALITIES[fmap_types.pop()]

            # Determine the name of the corresponding (first) magnitude file(s)
            magnitude = f"magnitude{'' if self.method == EstimatorType.MAPPED else '1'}"
            if magnitude not in suffix_set:
                try:
                    self.sources.append(
                        FieldmapFile(
                            sources[0]
                            .replace("fieldmap", "magnitude")
                            .replace("diff", "1")
                            .replace("phase", "magnitude")
                        )
                    )
                except Exception:
                    raise ValueError(
                        "A fieldmap or phase-difference estimation type was found, "
                        f"but an anatomical reference ({magnitude} file) is missing."
                    )

            # Check presence and try to find (if necessary) the second magnitude file
            if (
                self.method == EstimatorType.PHASEDIFF
                and "magnitude2" not in suffix_set
            ):
                try:
                    self.sources.append(
                        FieldmapFile(
                            sources[-1]
                            .replace("diff", "2")
                            .replace("phase", "magnitude")
                        )
                    )
                except Exception:
                    if "phase2" in suffix_set:
                        raise ValueError(
                            "A phase-difference estimation (phase1/2) type was found, "
                            "but an anatomical reference (magnitude2 file) is missing."
                        )

        # Fieldmap option 2: PEPOLAR (and fieldmap-less or ANAT)
        # IMPORTANT NOTE: fieldmap-less approaches can be considered PEPOLAR with RO = 0.0s
        pepolar_types = suffix_set.intersection(("bold", "dwi", "epi", "sbref"))
        anat_types = suffix_set.intersection(("T1w", "T2w"))
        _pepolar_estimation = (
            len([f for f in suffix_list if f in ("bold", "dwi", "epi", "sbref")]) > 1
        )

        if _pepolar_estimation and not anat_types:
            self.method = MODALITIES[pepolar_types.pop()]
            _pe = set(f.metadata["PhaseEncodingDirection"] for f in self.sources)
            if len(_pe) == 1:
                raise ValueError(
                    f"Only one phase-encoding direction <{_pe.pop()}> found across sources."
                )
        elif anat_types:
            self.method = MODALITIES[anat_types.pop()]

            if not pepolar_types:
                raise ValueError(
                    "Only anatomical sources were found, cannot estimate fieldmap."
                )

        if self.method == EstimatorType.UNKNOWN:
            # No method has been identified -> fail.
            raise ValueError("Insufficient sources to estimate a fieldmap.")

        intents_meta = set(
            el
            for f in self.sources
            for el in listify(f.metadata.get("IntendedFor") or [])
        )

        # Register this estimation method
        if not self.bids_id:
            # If not manually set, try to get it from BIDS metadata
            bids_ids = set(
                [
                    f.metadata.get("B0FieldIdentifier")
                    for f in self.sources
                    if f.metadata.get("B0FieldIdentifier")
                ]
            )
            if len(bids_ids) > 1:
                raise ValueError(
                    f"Multiple ``B0FieldIdentifier`` set: <{', '.join(bids_ids)}>"
                )
            elif bids_ids:
                object.__setattr__(self, "bids_id", bids_ids.pop())
            else:
                bids_id = _estimators.add(self.paths())
                object.__setattr__(self, "bids_id", bids_id)
                for intent_file in intents_meta:
                    _intents[intent_file].add(bids_id)
                return

        _estimators[self.bids_id] = self.paths()
        for intent_file in intents_meta:
            _intents[intent_file].add(self.bids_id)

    def paths(self):
        """Return a tuple of paths that are sorted."""
        return tuple(sorted(str(f.path) for f in self.sources))

    def get_workflow(self, **kwargs):
        """Build the estimation workflow corresponding to this instance."""
        if self._wf is not None:
            return self._wf

        # Override workflow name
        kwargs["name"] = f"wf_{self.bids_id}"

        if self.method in (EstimatorType.MAPPED, EstimatorType.PHASEDIFF):
            from .workflows.fit.fieldmap import init_fmap_wf

            kwargs["mode"] = str(self.method).rpartition(".")[-1].lower()
            self._wf = init_fmap_wf(**kwargs)
            self._wf.inputs.inputnode.magnitude = [
                str(f.path) for f in self.sources if f.suffix.startswith("magnitude")
            ]
            self._wf.inputs.inputnode.fieldmap = [
                (str(f.path), f.metadata)
                for f in self.sources
                if f.suffix in ("fieldmap", "phasediff", "phase2", "phase1")
            ]
        elif self.method == EstimatorType.PEPOLAR:
            from .workflows.fit.pepolar import init_topup_wf

            self._wf = init_topup_wf(**kwargs)
        elif self.method == EstimatorType.ANAT:
            from .workflows.fit.syn import init_syn_sdc_wf

            self._wf = init_syn_sdc_wf(**kwargs)

        return self._wf


def get_identifier(filename, by="IntendedFor"):
    """
    Return the fieldmap identifier for a given filename.

    Parameters
    ----------
    filename : :obj:`str`
        The file we want to search in indexes.
    by : :obj:`str` (``IntendedFor`` or ``sources``)
        Index table in which the search should be performed.

    """
    if by == "IntendedFor":
        return tuple(sorted(_intents.get(filename, ())))

    if by == "sources":
        return _estimators.get_key(filename)

    raise KeyError(r"'{by}'")


def clear_registry():
    """Empty registries."""
    _estimators.clear()
    _intents.clear()
