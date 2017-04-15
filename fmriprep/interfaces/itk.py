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
    CommandLine, CommandLineInputSpec, InputMultiPath, OutputMultiPath,
    isdefined)

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
    pe_dir = traits.Enum('i', 'i-', 'j', 'j-', 'k', 'k-',
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

        phaseEncDim = {'i': 0, 'j': 1, 'k': 2}[self.inputs.pe_dir[0]]

        if len(self.inputs.pe_dir) == 2:
            phaseEncSign = 1.0
        else:
            phaseEncSign = -1.0

        # Fix header
        hdr = nii.header.copy()
        hdr.set_data_dtype(np.dtype('<f4'))
        hdr.set_intent('vector', (), '')

        # Get data, convert to mm
        data = nii.get_data()

        aff = np.diag([1.0, 1.0, -1.0])
        if np.linalg.det(aff) < 0 and phaseEncDim != 0:
            # Reverse direction since ITK is LPS
            aff *= -1.0

        aff = aff.dot(nii.affine[:3, :3])

        data *= phaseEncSign * nii.header.get_zooms()[phaseEncDim]

        # Add missing dimensions
        zeros = np.zeros_like(data)
        field = [zeros, zeros]
        field.insert(phaseEncDim, data)
        field = np.stack(field, -1)
        # Add empty axis
        field = field[:, :, :, np.newaxis, :]

        # Write out
        self._results['out_file'] = genfname(
            self.inputs.in_file, suffix='antswarp')
        nb.Nifti1Image(
            field.astype(np.dtype('<f4')), nii.affine, hdr).to_filename(
                self._results['out_file'])

        return runtime


class AffineInitializerInputSpec(CommandLineInputSpec):
    dimension = traits.Enum(3, 2, usedefault=True, position=0, argstr='%s',
                            desc='dimension')
    fixed_image = File(exists=True, mandatory=True, position=1, argstr='%s',
                       desc='reference image')
    moving_image = File(exists=True, mandatory=True, position=2, argstr='%s',
                        desc='moving image')
    out_file = File('transform.mat', usedefault=True, position=3, argstr='%s',
                    desc='output transform file')
    # Defaults in antsBrainExtraction.sh -> 15 0.1 0 10
    search_factor = traits.Float(15.0, usedefault=True, position=4, argstr='%f',
                                 desc='increments (degrees) for affine search')
    radian_fraction = traits.Range(0.0, 1.0, value=0.1, usedefault=True, position=5,
                                   argstr='%f', desc='search this arc +/- principal axes')
    principal_axes = traits.Bool(
        False, usedefault=True, position=6, argstr='%d',
        desc='whether the rotation is searched around an initial principal axis alignment.')
    local_search = traits.Int(
        10, usedefault=True, position=7, argstr='%d',
        desc=' determines if a local optimization is run at each search point for the set '
             'number of iterations')


class AffineInitializerOutputSpec(TraitedSpec):
    out_file = File(desc='output transform file')


class AffineInitializer(CommandLine):
    """
    Initialize an affine transform (from antsBrainExtraction.sh)
    """
    _cmd = 'antsAffineInitializer'
    input_spec = AffineInitializerInputSpec
    output_spec = AffineInitializerOutputSpec

    def _list_outputs(self):
        return {'out_file': op.abspath(self.inputs.out_file)}
