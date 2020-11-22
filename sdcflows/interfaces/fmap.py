# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces to deal with the various types of fieldmap sources."""

from nipype import logging
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    traits,
    SimpleInterface,
)

LOGGER = logging.getLogger("nipype.interface")


class _PhaseMap2radsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input (wrapped) phase map")


class _PhaseMap2radsOutputSpec(TraitedSpec):
    out_file = File(desc="the phase map in the range 0 - 6.28")


class PhaseMap2rads(SimpleInterface):
    """Convert a phase map given in a.u. (e.g., 0-4096) to radians."""

    input_spec = _PhaseMap2radsInputSpec
    output_spec = _PhaseMap2radsOutputSpec

    def _run_interface(self, runtime):
        from ..utils.phasemanip import au2rads

        self._results["out_file"] = au2rads(self.inputs.in_file, newpath=runtime.cwd)
        return runtime


class _SubtractPhasesInputSpec(BaseInterfaceInputSpec):
    in_phases = traits.List(File(exists=True), min=1, max=2, desc="input phase maps")
    in_meta = traits.List(
        traits.Dict(), min=1, max=2, desc="metadata corresponding to the inputs"
    )


class _SubtractPhasesOutputSpec(TraitedSpec):
    phase_diff = File(exists=True, desc="phase difference map")
    metadata = traits.Dict(desc="output metadata")


class SubtractPhases(SimpleInterface):
    """Calculate a phase difference map."""

    input_spec = _SubtractPhasesInputSpec
    output_spec = _SubtractPhasesOutputSpec

    def _run_interface(self, runtime):
        if len(self.inputs.in_phases) != len(self.inputs.in_meta):
            raise ValueError(
                "Length of input phase-difference maps and metadata files "
                "should match."
            )

        if len(self.inputs.in_phases) == 1:
            self._results["phase_diff"] = self.inputs.in_phases[0]
            self._results["metadata"] = self.inputs.in_meta[0]
            return runtime

        from ..utils.phasemanip import subtract_phases as _subtract_phases

        # Discard in_meta traits with copy(), so that pop() works.
        self._results["phase_diff"], self._results["metadata"] = _subtract_phases(
            self.inputs.in_phases,
            (self.inputs.in_meta[0].copy(), self.inputs.in_meta[1].copy()),
            newpath=runtime.cwd,
        )

        return runtime


class _Phasediff2FieldmapInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="input fieldmap")
    metadata = traits.Dict(mandatory=True, desc="BIDS metadata dictionary")


class _Phasediff2FieldmapOutputSpec(TraitedSpec):
    out_file = File(desc="the output fieldmap")


class Phasediff2Fieldmap(SimpleInterface):
    """Convert a phase difference map into a fieldmap in Hz."""

    input_spec = _Phasediff2FieldmapInputSpec
    output_spec = _Phasediff2FieldmapOutputSpec

    def _run_interface(self, runtime):
        from ..utils.phasemanip import phdiff2fmap, delta_te as _delta_te

        self._results["out_file"] = phdiff2fmap(
            self.inputs.in_file, _delta_te(self.inputs.metadata), newpath=runtime.cwd
        )
        return runtime


# Code stub for #119 (B-Spline smoothing)
# class _Coefficients2WarpInputSpec(BaseInterfaceInputSpec):
#     in_target = File(exist=True, mandatory=True, desc="input EPI data to be corrected")
#     in_coeff = InputMultiObject(File(exists=True), mandatory=True,
#                                 desc="input coefficients, after alignment to the EPI data")
#     ro_time = traits.Float(mandatory=True, desc="EPI readout time")


# class _Coefficients2WarpOutputSpec(TraitedSpec):
#     out_warp = File(exists=True)


# class Coefficients2Warp(SimpleInterface):
#     """Convert coefficients to a full displacements map."""

#     input_spec = _Coefficients2WarpInputSpec
#     output_spec = _Coefficients2WarpOutputSpec

#     def _run_interface(self, runtime):
#         raise NotImplementedError
#         return runtime
