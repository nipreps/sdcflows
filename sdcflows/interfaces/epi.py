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
    File,
    InputMultiObject,
    OutputMultiObject,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)


class _GetReadoutTimeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="EPI image corresponding to the metadata")
    metadata = traits.Dict(mandatory=True, desc="metadata corresponding to the inputs")
    use_estimate = traits.Bool(
        False, usedefault=True, desc='Use "Estimated*" fields to calculate TotalReadoutTime'
    )
    fallback = traits.Float(desc="A fallback value, in seconds.")


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
            use_estimate=self.inputs.use_estimate,
            fallback=self.inputs.fallback or None,
        )
        self._results["pe_direction"] = self.inputs.metadata["PhaseEncodingDirection"]
        self._results["pe_dir_fsl"] = (
            self.inputs.metadata["PhaseEncodingDirection"]
            .replace("i", "x")
            .replace("j", "y")
            .replace("k", "z")
        )
        return runtime


class _SortPEBlipsInputSpec(BaseInterfaceInputSpec):
    in_data = InputMultiObject(
        File(exists=True),
        mandatory=True,
        desc="list of input data",
    )
    pe_dirs_fsl = InputMultiObject(
        traits.Enum("x", "x-", "y", "y-", "z", "z-"),
        mandatory=True,
        desc="list of PE directions, in FSL's conventions",
    )
    readout_times = InputMultiObject(
        traits.Float,
        mandatory=True,
        desc="list of total readout times"
    )


class _SortPEBlipsOutputSpec(TraitedSpec):
    out_data = OutputMultiObject(
        File(),
        desc="list of input data",
    )
    pe_dirs = OutputMultiObject(
        traits.Enum("i", "i-", "j", "j-", "k", "k-"),
        desc="list of PE directions, in BIDS's conventions",
    )
    pe_dirs_fsl = OutputMultiObject(
        traits.Enum("x", "x-", "y", "y-", "z", "z-"),
        desc="list of PE directions, in FSL's conventions",
    )
    readout_times = OutputMultiObject(
        traits.Float,
        desc="list of total readout times"
    )


class SortPEBlips(SimpleInterface):
    """Sort PE blips so they are consistently fed into TOPUP."""

    input_spec = _SortPEBlipsInputSpec
    output_spec = _SortPEBlipsOutputSpec

    def _run_interface(self, runtime):
        # Put sign first
        blips = [
            f"+{pe[0]}" if len(pe) == 1 else f"-{pe[0]}"
            for pe in self.inputs.pe_dirs_fsl
        ]
        sorted_inputs = sorted(zip(
            blips,
            self.inputs.readout_times,
            self.inputs.in_data,
        ))

        (
            self._results["pe_dirs_fsl"],
            self._results["readout_times"],
            self._results["out_data"],
        ) = zip(*sorted_inputs)

        # Put sign back last
        self._results["pe_dirs_fsl"] = [
            pe[1] if pe.startswith("+") else f"{pe[1]}-"
            for pe in self._results["pe_dirs_fsl"]
        ]
        self._results["pe_dirs"] = [
            pe.replace("x", "i").replace("y", "j").replace("z", "k")
            for pe in self._results["pe_dirs_fsl"]
        ]
        return runtime
