# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""Nipype interfaces wrapping :mod:`warpkit.api`.

`warpkit <https://github.com/vanandrew/warpkit>`__ implements MEDIC
(Multi-Echo DIstortion Correction). This module exposes the two MEDIC
stages SDCFlows actually drives — phase unwrapping and fieldmap
computation — as :class:`~nipype.interfaces.base.SimpleInterface`
subclasses calling :mod:`warpkit.api` in-process. ``warpkit`` is an
optional dependency; :class:`~nipype.interfaces.base.LibraryBaseInterface`
emits a clean "package not installed" error if it is missing at runtime.
"""

import os

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    InputMultiObject,
    LibraryBaseInterface,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    traits,
)

PE_DIRECTIONS = ('i', 'j', 'k', 'i-', 'j-', 'k-', 'x', 'y', 'z', 'x-', 'y-', 'z-')


class WarpkitBaseInterface(LibraryBaseInterface):
    """Base for all warpkit-backed interfaces."""

    _pkg = 'warpkit'


# ---------------------------------------------------------------------------
# UnwrapPhase — ROMEO multi-echo phase unwrapping
# ---------------------------------------------------------------------------


class _UnwrapPhaseInputSpec(BaseInterfaceInputSpec):
    phase = InputMultiObject(File(exists=True), mandatory=True)
    magnitude = InputMultiObject(File(exists=True), mandatory=True)
    echo_times = traits.List(traits.Float, xor=['metadata'])
    metadata = InputMultiObject(File(exists=True), xor=['echo_times'])
    out_prefix = traits.Str('unwrap', usedefault=True)
    num_threads = traits.Int(
        1, usedefault=True, nohash=True, desc='Number of threads to use for unwrapping'
    )
    wrap_limit = traits.Bool(False, usedefault=True)
    debug = traits.Bool(False, usedefault=True)


class _UnwrapPhaseOutputSpec(TraitedSpec):
    unwrapped = OutputMultiObject(File(exists=True), desc='unwrapped phase per echo')
    masks = File(exists=True, desc='per-frame masks NIfTI')


class UnwrapPhase(WarpkitBaseInterface, SimpleInterface):
    """ROMEO multi-echo phase unwrapping (:func:`warpkit.api.unwrap_phase`)."""

    input_spec = _UnwrapPhaseInputSpec
    output_spec = _UnwrapPhaseOutputSpec

    def _run_interface(self, runtime):
        from warpkit.api import unwrap_phase

        out_prefix = os.path.join(runtime.cwd, self.inputs.out_prefix)
        result = unwrap_phase(
            phase=self.inputs.phase,
            magnitude=self.inputs.magnitude,
            out_prefix=out_prefix,
            tes=self.inputs.echo_times or None,
            metadata=self.inputs.metadata or None,
            n_cpus=self.inputs.num_threads,
            wrap_limit=self.inputs.wrap_limit,
            debug=self.inputs.debug,
        )

        self._results['unwrapped'] = [str(p) for p in result.unwrapped]
        self._results['masks'] = str(result.masks)
        return runtime


# ---------------------------------------------------------------------------
# ComputeFieldmap — post-unwrap stage of MEDIC
# ---------------------------------------------------------------------------


class _ComputeFieldmapInputSpec(BaseInterfaceInputSpec):
    unwrapped = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc='unwrapped phase per echo (output of UnwrapPhase)',
    )
    magnitude = InputMultiObject(File(exists=True), mandatory=True)
    masks = File(exists=True, mandatory=True, desc='per-frame masks (output of UnwrapPhase)')
    echo_times = traits.List(traits.Float, xor=['metadata'])
    total_readout_time = traits.Float(xor=['metadata'])
    phase_encoding_direction = traits.Enum(*PE_DIRECTIONS, xor=['metadata'])
    metadata = InputMultiObject(
        File(exists=True),
        xor=['echo_times', 'total_readout_time', 'phase_encoding_direction'],
    )
    out_prefix = traits.Str('fieldmap', usedefault=True)
    # NOTE: `traits.Tuple(Int(), Int(), default=(1, 5))` is silently ignored —
    # the inner Int()s default to 0, and the outer `default` kwarg loses.
    # That collapses the border-filter to 0 SVD components, which zeros the
    # mask==1 ring in warpkit.unwrap.svd_filtering and makes the dynamic
    # fieldmap appear hard-brain-masked. Pass defaults to the inner Ints.
    border_filt = traits.Tuple(
        traits.Int(1),
        traits.Int(5),
        usedefault=True,
        desc='SVD components for the two-pass border filter',
    )
    svd_filt = traits.Int(10, usedefault=True, desc='Number of singular values to truncate to')
    num_threads = traits.Int(
        1,
        usedefault=True,
        nohash=True,
        desc='Number of threads to use for estimating the fieldmap',
    )


class _ComputeFieldmapOutputSpec(TraitedSpec):
    fieldmap_native = File(exists=True)
    displacement_map = File(exists=True)
    fieldmap = File(exists=True)


class ComputeFieldmap(WarpkitBaseInterface, SimpleInterface):
    """Post-unwrap MEDIC stage (:func:`warpkit.api.compute_fieldmap`)."""

    input_spec = _ComputeFieldmapInputSpec
    output_spec = _ComputeFieldmapOutputSpec

    def _run_interface(self, runtime):
        from warpkit.api import compute_fieldmap

        out_prefix = os.path.join(runtime.cwd, self.inputs.out_prefix)
        result = compute_fieldmap(
            unwrapped=self.inputs.unwrapped,
            magnitude=self.inputs.magnitude,
            masks=self.inputs.masks,
            out_prefix=out_prefix,
            tes=self.inputs.echo_times or None,
            total_readout_time=self.inputs.total_readout_time or None,
            phase_encoding_direction=self.inputs.phase_encoding_direction or None,
            metadata=self.inputs.metadata or None,
            border_filt=self.inputs.border_filt,
            svd_filt=self.inputs.svd_filt,
            n_cpus=self.inputs.num_threads,
        )

        self._results['fieldmap_native'] = str(result.fieldmap_native)
        self._results['displacement_map'] = str(result.displacement_map)
        self._results['fieldmap'] = str(result.fieldmap)
        return runtime
