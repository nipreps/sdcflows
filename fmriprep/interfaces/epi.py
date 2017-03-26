#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Convenience tools for handling :abbr:`EPI (echo planar imaging)` images
"""

import os
import numpy as np
import os.path as op
from nipype.interfaces.base import (traits, isdefined, TraitedSpec, BaseInterface,
                                    BaseInterfaceInputSpec, File, InputMultiPath,
                                    OutputMultiPath)
from nilearn.image import mean_img
from fmriprep.utils.misc import genfname
from builtins import str, bytes


class SelectReferenceInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='input files')
    reference = File(exists=True, desc='reference file')
    njobs = traits.Int(1, nohash=True, usedefault=True, desc='number of jobs')

class SelectReferenceOutputSpec(TraitedSpec):
    reference = File(exists=True, desc='reference image')

class SelectReference(BaseInterface):
    """
    If ``reference`` is set, forwards it to the output. Computes the time-average of
    ``in_files`` otherwise
    """

    input_spec = SelectReferenceInputSpec
    output_spec = SelectReferenceOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(SelectReference, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        # Return reference
        if isdefined(self.inputs.reference):
            self._results['reference'] = self.inputs.reference
            return runtime

        in_files = self.inputs.in_files
        if isinstance(in_files, (str, bytes)):
            in_files = [in_files]

        if len(in_files) == 1:
            self._results['reference'] = in_files[0]
            return runtime

        # or mean otherwise
        self._results['reference'] = genfname(in_files[0], suffix='avg')
        mean_img(in_files, n_jobs=self.inputs.njobs).to_filename(
            self._results['reference'])
        return runtime

