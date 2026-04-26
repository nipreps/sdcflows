# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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
(Multi-Echo DIstortion Correction) and related warp utilities. Each of
warpkit's seven ``wk-*`` CLI tools is mirrored here as a
:class:`~nipype.interfaces.base.SimpleInterface`, calling the corresponding
:mod:`warpkit.api` function in-process. ``warpkit`` is an optional dependency
— :class:`~nipype.interfaces.base.LibraryBaseInterface` produces a clean
"package not installed" error if ``warpkit`` is missing at runtime.
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
    isdefined,
    traits,
)

PE_DIRECTIONS = ('i', 'j', 'k', 'i-', 'j-', 'k-', 'x', 'y', 'z', 'x-', 'y-', 'z-')
PE_AXES = ('i', 'j', 'k', 'x', 'y', 'z')
WARP_FORMATS = ('itk', 'fsl', 'ants', 'afni')


class WarpkitBaseInterface(LibraryBaseInterface):
    """Base for all warpkit-backed interfaces."""

    _pkg = 'warpkit'


def _as_str_list(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    return [str(p) for p in x]


# ---------------------------------------------------------------------------
# MEDIC — full multi-echo distortion correction pipeline
# ---------------------------------------------------------------------------


class _MEDICInputSpec(BaseInterfaceInputSpec):
    phase = InputMultiObject(
        File(exists=True), mandatory=True, desc='phase NIfTI, one per echo'
    )
    magnitude = InputMultiObject(
        File(exists=True), mandatory=True, desc='magnitude NIfTI, one per echo'
    )
    echo_times = traits.List(
        traits.Float,
        xor=['metadata'],
        desc='echo times in milliseconds, one per echo',
    )
    total_readout_time = traits.Float(
        xor=['metadata'], desc='EPI total readout time in seconds'
    )
    phase_encoding_direction = traits.Enum(
        *PE_DIRECTIONS,
        xor=['metadata'],
        desc='phase-encoding direction (with sign)',
    )
    metadata = InputMultiObject(
        File(exists=True),
        xor=['echo_times', 'total_readout_time', 'phase_encoding_direction'],
        desc='BIDS sidecar JSONs, one per echo (alternative to direct args)',
    )
    out_prefix = traits.Str('medic', usedefault=True, desc='prefix for output filenames')
    noise_frames = traits.Int(
        0, usedefault=True, desc='number of trailing noise frames to drop'
    )
    n_cpus = traits.Int(4, usedefault=True, desc='number of CPUs to use')
    wrap_limit = traits.Bool(
        False, usedefault=True, desc='disable some phase-unwrapping heuristics'
    )
    debug = traits.Bool(False, usedefault=True, desc='enable debug mode')


class _MEDICOutputSpec(TraitedSpec):
    fieldmap_native = File(exists=True, desc='native-space B0 field map (Hz)')
    displacement_map = File(exists=True, desc='displacement map (mm)')
    fieldmap = File(exists=True, desc='undistorted-space B0 field map (Hz)')


class MEDIC(WarpkitBaseInterface, SimpleInterface):
    """Run the full MEDIC pipeline (:func:`warpkit.api.medic`)."""

    input_spec = _MEDICInputSpec
    output_spec = _MEDICOutputSpec

    def _run_interface(self, runtime):
        from warpkit.api import medic

        out_prefix = os.path.join(runtime.cwd, self.inputs.out_prefix)
        try:
            result = medic(
                phase=list(self.inputs.phase),
                magnitude=list(self.inputs.magnitude),
                out_prefix=out_prefix,
                tes=(
                    list(self.inputs.echo_times)
                    if isdefined(self.inputs.echo_times)
                    else None
                ),
                total_readout_time=(
                    self.inputs.total_readout_time
                    if isdefined(self.inputs.total_readout_time)
                    else None
                ),
                phase_encoding_direction=(
                    self.inputs.phase_encoding_direction
                    if isdefined(self.inputs.phase_encoding_direction)
                    else None
                ),
                metadata=(
                    list(self.inputs.metadata)
                    if isdefined(self.inputs.metadata)
                    else None
                ),
                noise_frames=self.inputs.noise_frames,
                n_cpus=self.inputs.n_cpus,
                wrap_limit=self.inputs.wrap_limit,
                debug=self.inputs.debug,
            )
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        self._results['fieldmap_native'] = str(result.fieldmap_native)
        self._results['displacement_map'] = str(result.displacement_map)
        self._results['fieldmap'] = str(result.fieldmap)
        return runtime


# ---------------------------------------------------------------------------
# UnwrapPhase — ROMEO multi-echo phase unwrapping
# ---------------------------------------------------------------------------


class _UnwrapPhaseInputSpec(BaseInterfaceInputSpec):
    phase = InputMultiObject(File(exists=True), mandatory=True)
    magnitude = InputMultiObject(File(exists=True), mandatory=True)
    echo_times = traits.List(traits.Float, xor=['metadata'])
    metadata = InputMultiObject(File(exists=True), xor=['echo_times'])
    out_prefix = traits.Str('unwrap', usedefault=True)
    noise_frames = traits.Int(0, usedefault=True)
    n_cpus = traits.Int(4, usedefault=True)
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
        try:
            result = unwrap_phase(
                phase=list(self.inputs.phase),
                magnitude=list(self.inputs.magnitude),
                out_prefix=out_prefix,
                tes=(
                    list(self.inputs.echo_times)
                    if isdefined(self.inputs.echo_times)
                    else None
                ),
                metadata=(
                    list(self.inputs.metadata)
                    if isdefined(self.inputs.metadata)
                    else None
                ),
                noise_frames=self.inputs.noise_frames,
                n_cpus=self.inputs.n_cpus,
                wrap_limit=self.inputs.wrap_limit,
                debug=self.inputs.debug,
            )
        except ValueError as e:
            raise RuntimeError(str(e)) from e

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
    masks = File(
        exists=True, mandatory=True, desc='per-frame masks (output of UnwrapPhase)'
    )
    echo_times = traits.List(traits.Float, xor=['metadata'])
    total_readout_time = traits.Float(xor=['metadata'])
    phase_encoding_direction = traits.Enum(*PE_DIRECTIONS, xor=['metadata'])
    metadata = InputMultiObject(
        File(exists=True),
        xor=['echo_times', 'total_readout_time', 'phase_encoding_direction'],
    )
    out_prefix = traits.Str('fieldmap', usedefault=True)
    border_filt = traits.Tuple(
        traits.Int(),
        traits.Int(),
        default=(1, 5),
        usedefault=True,
        desc='SVD components for the two-pass border filter',
    )
    svd_filt = traits.Int(10, usedefault=True)
    n_cpus = traits.Int(4, usedefault=True)


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
        try:
            result = compute_fieldmap(
                unwrapped=list(self.inputs.unwrapped),
                magnitude=list(self.inputs.magnitude),
                masks=self.inputs.masks,
                out_prefix=out_prefix,
                tes=(
                    list(self.inputs.echo_times)
                    if isdefined(self.inputs.echo_times)
                    else None
                ),
                total_readout_time=(
                    self.inputs.total_readout_time
                    if isdefined(self.inputs.total_readout_time)
                    else None
                ),
                phase_encoding_direction=(
                    self.inputs.phase_encoding_direction
                    if isdefined(self.inputs.phase_encoding_direction)
                    else None
                ),
                metadata=(
                    list(self.inputs.metadata)
                    if isdefined(self.inputs.metadata)
                    else None
                ),
                border_filt=tuple(self.inputs.border_filt),
                svd_filt=self.inputs.svd_filt,
                n_cpus=self.inputs.n_cpus,
            )
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        self._results['fieldmap_native'] = str(result.fieldmap_native)
        self._results['displacement_map'] = str(result.displacement_map)
        self._results['fieldmap'] = str(result.fieldmap)
        return runtime


# ---------------------------------------------------------------------------
# ApplyWarp — resample through a displacement transform
# ---------------------------------------------------------------------------


class _ApplyWarpInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    transform = InputMultiObject(File(exists=True), mandatory=True)
    out_file = traits.Str(desc='output path; defaults to <cwd>/applied.nii.gz')
    transform_type = traits.Enum('map', 'field', mandatory=True)
    reference = File(exists=True)
    phase_encoding_axis = traits.Enum(*PE_AXES)
    format = traits.Enum(*WARP_FORMATS, usedefault=True, default='itk')


class _ApplyWarpOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class ApplyWarp(WarpkitBaseInterface, SimpleInterface):
    """Resample an image through a warpkit displacement transform
    (:func:`warpkit.api.apply_warp`)."""

    input_spec = _ApplyWarpInputSpec
    output_spec = _ApplyWarpOutputSpec

    def _run_interface(self, runtime):
        from warpkit.api import apply_warp

        out_file = self.inputs.out_file
        if not isdefined(out_file):
            out_file = os.path.join(runtime.cwd, 'applied.nii.gz')
        try:
            result = apply_warp(
                input=self.inputs.in_file,
                transform=list(self.inputs.transform),
                output=out_file,
                transform_type=self.inputs.transform_type,
                reference=(
                    self.inputs.reference
                    if isdefined(self.inputs.reference)
                    else None
                ),
                phase_encoding_axis=(
                    self.inputs.phase_encoding_axis
                    if isdefined(self.inputs.phase_encoding_axis)
                    else None
                ),
                format=self.inputs.format,
            )
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        self._results['out_file'] = str(result.output)
        return runtime


# ---------------------------------------------------------------------------
# ConvertWarp — interconvert / reformat / invert displacement transforms
# ---------------------------------------------------------------------------


class _ConvertWarpInputSpec(BaseInterfaceInputSpec):
    in_file = InputMultiObject(File(exists=True), mandatory=True)
    out_file = traits.Either(
        traits.Str(),
        traits.List(traits.Str()),
        desc='output path(s); 1 path bundles, N paths split per frame',
    )
    from_type = traits.Enum('map', 'field', mandatory=True)
    to_type = traits.Enum('map', 'field')
    from_format = traits.Enum(*WARP_FORMATS, usedefault=True, default='itk')
    to_format = traits.Enum(*WARP_FORMATS, usedefault=True, default='itk')
    axis = traits.Enum(*PE_AXES)
    frame = traits.Int()
    invert = traits.Bool(False, usedefault=True)
    verbose = traits.Bool(False, usedefault=True)


class _ConvertWarpOutputSpec(TraitedSpec):
    out_file = OutputMultiObject(File(exists=True))


class ConvertWarp(WarpkitBaseInterface, SimpleInterface):
    """Convert displacement transforms (:func:`warpkit.api.convert_warp`)."""

    input_spec = _ConvertWarpInputSpec
    output_spec = _ConvertWarpOutputSpec

    def _run_interface(self, runtime):
        from warpkit.api import convert_warp

        if isdefined(self.inputs.out_file):
            out_paths = _as_str_list(self.inputs.out_file)
        else:
            out_paths = [os.path.join(runtime.cwd, 'converted.nii.gz')]

        try:
            result = convert_warp(
                input=list(self.inputs.in_file),
                output=out_paths,
                from_type=self.inputs.from_type,
                to_type=self.inputs.to_type if isdefined(self.inputs.to_type) else None,
                from_format=self.inputs.from_format,
                to_format=self.inputs.to_format,
                axis=self.inputs.axis if isdefined(self.inputs.axis) else None,
                frame=self.inputs.frame if isdefined(self.inputs.frame) else None,
                invert=self.inputs.invert,
                verbose=self.inputs.verbose,
            )
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        self._results['out_file'] = [str(p) for p in result.output]
        return runtime


# ---------------------------------------------------------------------------
# ConvertFieldmap — mm displacement <-> Hz fieldmap
# ---------------------------------------------------------------------------


class _ConvertFieldmapInputSpec(BaseInterfaceInputSpec):
    in_file = InputMultiObject(File(exists=True), mandatory=True)
    out_file = traits.Either(
        traits.Str(),
        traits.List(traits.Str()),
    )
    from_type = traits.Enum('map', 'field', 'fieldmap', mandatory=True)
    to_type = traits.Enum('map', 'field', 'fieldmap', mandatory=True)
    total_readout_time = traits.Float(mandatory=True)
    phase_encoding_direction = traits.Enum(*PE_AXES, mandatory=True)
    from_format = traits.Enum(*WARP_FORMATS, usedefault=True, default='itk')
    to_format = traits.Enum(*WARP_FORMATS, usedefault=True, default='itk')
    flip_sign = traits.Bool(False, usedefault=True)
    frame = traits.Int()


class _ConvertFieldmapOutputSpec(TraitedSpec):
    out_file = OutputMultiObject(File(exists=True))


class ConvertFieldmap(WarpkitBaseInterface, SimpleInterface):
    """Convert between mm displacement and Hz fieldmap
    (:func:`warpkit.api.convert_fieldmap`)."""

    input_spec = _ConvertFieldmapInputSpec
    output_spec = _ConvertFieldmapOutputSpec

    def _run_interface(self, runtime):
        from warpkit.api import convert_fieldmap

        if isdefined(self.inputs.out_file):
            out_paths = _as_str_list(self.inputs.out_file)
        else:
            out_paths = [os.path.join(runtime.cwd, 'converted.nii.gz')]

        try:
            result = convert_fieldmap(
                input=list(self.inputs.in_file),
                output=out_paths,
                from_type=self.inputs.from_type,
                to_type=self.inputs.to_type,
                total_readout_time=self.inputs.total_readout_time,
                phase_encoding_direction=self.inputs.phase_encoding_direction,
                from_format=self.inputs.from_format,
                to_format=self.inputs.to_format,
                flip_sign=self.inputs.flip_sign,
                frame=self.inputs.frame if isdefined(self.inputs.frame) else None,
            )
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        self._results['out_file'] = [str(p) for p in result.output]
        return runtime


# ---------------------------------------------------------------------------
# ComputeJacobian
# ---------------------------------------------------------------------------


class _ComputeJacobianInputSpec(BaseInterfaceInputSpec):
    in_file = InputMultiObject(File(exists=True), mandatory=True)
    out_file = traits.Either(
        traits.Str(),
        traits.List(traits.Str()),
    )
    from_type = traits.Enum('map', 'field', mandatory=True)
    from_format = traits.Enum(*WARP_FORMATS, usedefault=True, default='itk')
    axis = traits.Enum(*PE_AXES)
    frame = traits.Int()


class _ComputeJacobianOutputSpec(TraitedSpec):
    out_file = OutputMultiObject(File(exists=True))


class ComputeJacobian(WarpkitBaseInterface, SimpleInterface):
    """Jacobian determinant of a displacement warp
    (:func:`warpkit.api.compute_jacobian`)."""

    input_spec = _ComputeJacobianInputSpec
    output_spec = _ComputeJacobianOutputSpec

    def _run_interface(self, runtime):
        from warpkit.api import compute_jacobian

        if isdefined(self.inputs.out_file):
            out_paths = _as_str_list(self.inputs.out_file)
        else:
            out_paths = [os.path.join(runtime.cwd, 'jacobian.nii.gz')]

        try:
            result = compute_jacobian(
                input=list(self.inputs.in_file),
                output=out_paths,
                from_type=self.inputs.from_type,
                from_format=self.inputs.from_format,
                axis=self.inputs.axis if isdefined(self.inputs.axis) else None,
                frame=self.inputs.frame if isdefined(self.inputs.frame) else None,
            )
        except ValueError as e:
            raise RuntimeError(str(e)) from e

        self._results['out_file'] = [str(p) for p in result.output]
        return runtime
