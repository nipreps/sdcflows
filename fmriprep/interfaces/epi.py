#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Convenience tools for handling :abbr:`EPI (echo planar imaging)` images
"""

import os
import os.path as op

import nibabel as nb
import numpy as np
from nilearn.image import mean_img

from nipype.interfaces.base import (traits, isdefined, TraitedSpec, BaseInterface,
                                    BaseInterfaceInputSpec, File, InputMultiPath,
                                    OutputMultiPath)
from nipype.algorithms.confounds import is_outlier

from nipype.interfaces.afni import Volreg
from fmriprep.interfaces.bids import SimpleInterface
from fmriprep.utils.misc import genfname
from builtins import str, bytes


class EstimateReferenceImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="4D EPI file")


class EstimateReferenceImageOutputSpec(TraitedSpec):
    ref_image = File(exists=True, desc="3D reference image")
    n_volumes_to_discard = traits.Int(desc="Number of detected non-steady "
                                           "state volumes in the beginning of "
                                           "the input file")


class EstimateReferenceImage(SimpleInterface):
    """
    Given an 4D EPI file estimate an optimal reference image that could be later
    used for motion estimation and coregistration purposes. If detected uses
    T1 saturated volumes (non-steady state) otherwise a median of
    of a subset of motion corrected volumes is used.
    """
    input_spec = EstimateReferenceImageInputSpec
    output_spec = EstimateReferenceImageOutputSpec

    def _run_interface(self, runtime):
        in_nii = nb.load(self.inputs.in_file)
        global_signal = in_nii.get_data()[:, :, :, :50].mean(axis=0).mean(
            axis=0).mean(axis=0)

        n_volumes_to_discard = is_outlier(global_signal)

        out_ref_fname = os.path.abspath("ref_image.nii.gz")

        if n_volumes_to_discard == 0:
            if in_nii.shape[-1] > 40:
                slice = in_nii.get_data()[:, :, :, 20:40]
                slice_fname = os.path.abspath("slice.nii.gz")
                nb.Nifti1Image(slice, in_nii.affine,
                               in_nii.header).to_filename(slice_fname)
            else:
                slice_fname = self.inputs.in_file

            res = Volreg(in_file=slice_fname, args='-Fourier -twopass', zpad=4,
                         outputtype='NIFTI_GZ').run()

            mc_slice_nii = nb.load(res.outputs.out_file)

            median_image_data = np.median(mc_slice_nii.get_data(), axis=3)
            nb.Nifti1Image(median_image_data, mc_slice_nii.affine,
                           mc_slice_nii.header).to_filename(out_ref_fname)
        else:
            median_image_data = np.median(
                in_nii.get_data()[:, :, :, :n_volumes_to_discard], axis=3)
            nb.Nifti1Image(median_image_data, in_nii.affine,
                           in_nii.header).to_filename(out_ref_fname)

        self._results["ref_image"] = out_ref_fname
        self._results["n_volumes_to_discard"] = n_volumes_to_discard

        return runtime


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

