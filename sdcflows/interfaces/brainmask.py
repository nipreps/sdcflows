# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Brain extraction interfaces."""
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    SimpleInterface,
)
from ..utils.tools import brain_masker


class _BrainExtractionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="file to mask")


class _BrainExtractionOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="the input file, after masking")
    out_mask = File(exists=True, desc="the binary brain mask")
    out_probseg = File(exists=True, desc="the probabilistic brain mask")


class BrainExtraction(SimpleInterface):
    """Brain extraction for EPI and GRE data."""

    input_spec = _BrainExtractionInputSpec
    output_spec = _BrainExtractionOutputSpec

    def _run_interface(self, runtime):
        (
            self._results["out_file"],
            self._results["out_probseg"],
            self._results["out_mask"],
        ) = brain_masker(
            self.inputs.in_file,
            fname_presuffix(self.inputs.in_file, suffix="_mask", newpath=runtime.cwd),
        )
        return runtime
