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
    pe_dir_fsl = traits.Enum("x", "x-", "y", "y-", "z", "z-")


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
        self._results["pe_dir_fsl"] = (
            self.inputs.metadata["PhaseEncodingDirection"]
            .replace("i", "x")
            .replace("j", "y")
            .replace("k", "z")
        )
        return runtime
