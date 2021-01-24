"""Interfaces to deal with the various types of fieldmap sources."""

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    isdefined,
    traits,
    SimpleInterface,
)


class _GetReadoutTimeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="EPI image corresponding to the metadata")
    metadata = traits.Dict(mandatory=True, desc="metadata corresponding to the inputs")


class _GetReadoutTimeOutputSpec(TraitedSpec):
    readout_time = traits.Float
    pe_direction = traits.Enum("i", "i-", "j", "j-", "k", "k-")


class GetReadoutTime(SimpleInterface):
    """Calculate the readout time from available metadata."""

    input_spec = _GetReadoutTimeInputSpec
    output_spec = _GetReadoutTimeOutputSpec

    def _run_interface(self, runtime):
        from ..utils.epimanip import get_trt

        self._results["readout_time"] = get_trt(
            self.inputs.metadata,
            self.inputs.in_file if isdefined(self.inputs.in_file) else None,
        )
        self._results["pe_direction"] = self.inputs.metadata["PhaseEncodingDirection"]
        return runtime
