#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
ITK files handling
~~~~~~~~~~~~~~~~~~


"""
import os
import numpy as np
import nibabel as nb

from joblib import Parallel, delayed
from niworkflows.nipype.utils.filemanip import fname_presuffix
from niworkflows.nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, InputMultiPath)

from niworkflows.interfaces.base import SimpleInterface


class MCFLIRT2ITKInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True,
                              desc='list of MAT files from MCFLIRT')
    in_reference = File(exists=True, mandatory=True,
                        desc='input image for spatial reference')
    in_source = File(exists=True, mandatory=True,
                     desc='input image for spatial source')
    nprocs = traits.Int(1, usedefault=True, nohash=True,
                        desc='number of parallel processes')


class MCFLIRT2ITKOutputSpec(TraitedSpec):
    out_file = File(desc='the output ITKTransform file')


class MCFLIRT2ITK(SimpleInterface):

    """
    Convert a list of MAT files from MCFLIRT into an ITK Transform file.
    """
    input_spec = MCFLIRT2ITKInputSpec
    output_spec = MCFLIRT2ITKOutputSpec

    def _run_interface(self, runtime):

        parallel = Parallel(n_jobs=self.inputs.nprocs)
        itk_outs = parallel(delayed(_mat2itk)(
            in_mat, self.inputs.in_reference, self.inputs.in_source,
            i, runtime.cwd) for i, in_mat in enumerate(self.inputs.in_files))

        # Compose the collated ITK transform file and write
        tfms = '#Insight Transform File V1.0\n' + ''.join(
            [el[1] for el in sorted(itk_outs)])

        self._results['out_file'] = os.path.abspath('mat2itk.txt')
        with open(self._results['out_file'], 'w') as f:
            f.write(tfms)

        return runtime


class FUGUEvsm2ANTSwarpInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='input displacements field map')
    pe_dir = traits.Enum('i', 'i-', 'j', 'j-', 'k', 'k-',
                         desc='phase-encoding axis')


class FUGUEvsm2ANTSwarpOutputSpec(TraitedSpec):
    out_file = File(desc='the output warp field')


class FUGUEvsm2ANTSwarp(SimpleInterface):

    """
    Convert a voxel-shift-map to ants warp

    """
    input_spec = FUGUEvsm2ANTSwarpInputSpec
    output_spec = FUGUEvsm2ANTSwarpOutputSpec

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
        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_antswarp', newpath=runtime.cwd)
        nb.Nifti1Image(
            field.astype(np.dtype('<f4')), nii.affine, hdr).to_filename(
                self._results['out_file'])

        return runtime


def _mat2itk(in_file, in_ref, in_src, index=0, newpath=None):
    from niworkflows.nipype.interfaces.c3 import C3dAffineTool
    from niworkflows.nipype.utils.filemanip import fname_presuffix

    # Generate a temporal file name
    out_file = fname_presuffix(in_file, suffix='_itk-%05d.txt' % index,
                               newpath=newpath)

    # Run c3d_affine_tool
    C3dAffineTool(transform_file=in_file, reference_file=in_ref, source_file=in_src,
                  fsl2ras=True, itk_transform=out_file).run()
    transform = '#Transform %d\n' % index
    with open(out_file) as itkfh:
        transform += ''.join(itkfh.readlines()[2:])

    return (index, transform)
