#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals
from os import path as op
import numpy as np
import nibabel as nb
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, File,
    InputMultiPath, OutputMultiPath, isdefined
)

from io import open
from fmriprep.utils.misc import genfname

ITK_TFM_HEADER = "#Insight Transform File V1.0"
ITK_TFM_TPL = """\
#Transform {tf_id}
Transform: {tf_type}
Parameters: {tf_params}
FixedParameters: {fixed_params}""".format


class MergeANTsTransformsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc='input file')
    in_file_invert = traits.Bool(False, usedefault=True)
    position = traits.Int(-1, usedefault=True)
    transforms = InputMultiPath(File(exists=True),
                                mandatory=True, desc='input file')
    invert_transform_flags = traits.List(traits.Bool(), desc='invert transforms')

class MergeANTsTransformsOutputSpec(TraitedSpec):
    transforms = OutputMultiPath(File(exists=True),
                                 desc='list of output files')
    invert_transform_flags = traits.List(
        traits.Bool(), desc='invert transforms')

class MergeANTsTransforms(BaseInterface):

    """
    This interface generates an identity transform if the input
    is not set.

    """
    input_spec = MergeANTsTransformsInputSpec
    output_spec = MergeANTsTransformsOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(MergeANTsTransforms, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        self._results['transforms'] = self.inputs.transforms

        self._results['invert_transform_flags'] = [False] * len(self.inputs.transforms)
        if isdefined(self.inputs.invert_transform_flags):
            self._results['invert_transform_flags'] = self.inputs.invert_transform_flags

        if isdefined(self.inputs.in_file) and self.inputs.in_file is not None:
            flag = self.inputs.in_file_invert
            in_file = self.inputs.in_file
            pos = self.inputs.position
            if pos == -1:
                self._results['transforms'] += [in_file]
                self._results['invert_transform_flags'] += [flag]
            else:
                self._results['transforms'].insert(pos, in_file)
                self._results['invert_transform_flags'].insert(pos, flag)

        return runtime


class FUGUEvsm2ANTSwarpInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='input displacements field map')
    pe_dir = traits.Enum('y', 'y-', 'x', 'x-', usedefault=True,
                         desc='phase-encoding axis')
    units = traits.Enum('vox', 'mm', usedefault=True,
                        desc='units of the input field')

class FUGUEvsm2ANTSwarpOutputSpec(TraitedSpec):
    out_file = File(desc='the output warp field')


class FUGUEvsm2ANTSwarp(BaseInterface):

    """
    Convert a voxel-shift-map to ants warp

    """
    input_spec = FUGUEvsm2ANTSwarpInputSpec
    output_spec = FUGUEvsm2ANTSwarpOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(FUGUEvsm2ANTSwarp, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):

        nii = nb.load(self.inputs.in_file)

        pe_dir = 1 if 'y' in self.inputs.pe_dir else 0

        # Fix header
        hdr = nii.header.copy()
        hdr.set_data_dtype(np.dtype('<f4'))
        hdr.set_intent('vector', (), '')

        # Get data, convert to mm
        data = nii.get_data()

        aff = nii.affine
        if np.linalg.det(aff) < 0:
            # Reverse direction since ITK is LPS
            aff = np.diag([-1, -1, 1, 1]).dot(aff)
            data *= -1.0

        if self.inputs.units == 'vox':
            spacing = hdr.get_zooms()[pe_dir]
            data *= spacing

        # Add missing dimensions
        zeros = np.zeros_like(data)
        field = [zeros, zeros]
        field.insert(pe_dir, data)
        field = np.stack(field, -1)
        # Add empty axis
        field = field[:, :, :, np.newaxis, :]

        # Write out
        self._results['out_file'] = genfname(
            self.inputs.in_file, suffix='antswarp')
        nb.Nifti1Image(
            field.astype(np.dtype('<f4')), aff, hdr).to_filename(
                self._results['out_file'])

        return runtime
