#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nb
import scipy.ndimage as nd
from nilearn.image import resample_to_img

from niworkflows.nipype.utils.filemanip import fname_presuffix
from niworkflows.nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, InputMultiPath
)
from niworkflows.interfaces.base import SimpleInterface


class ApplyMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    in_mask = File(exists=True, mandatory=True, desc='input mask')


class ApplyMaskOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output average file')


class ApplyMask(SimpleInterface):
    input_spec = ApplyMaskInputSpec
    output_spec = ApplyMaskOutputSpec

    def _run_interface(self, runtime):
        out_file = fname_presuffix(self.inputs.in_file, suffix='_brainmask',
                                   newpath=runtime.cwd)
        nii = nb.load(self.inputs.in_file)
        data = nii.get_data()
        data[nb.load(self.inputs.in_mask).get_data() <= 0] = 0
        nb.Nifti1Image(data, nii.affine, nii.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime


class TPM2ROIInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    in_mask = File(exists=True, mandatory=True, desc='input mask')
    mask_erode_mm = traits.Float(0.0, usedefault=True,
                                 desc='erode input mask (kernel width in mm)')
    erode_mm = traits.Float(0.0, usedefault=True,
                            desc='erode output mask (kernel width in mm)')
    prob_thresh = traits.Float(0.95, usedefault=True,
                               desc='threshold for the tissue probability maps')


class TPM2ROIOutputSpec(TraitedSpec):
    roi_file = File(exists=True, desc='output ROI file')
    eroded_mask = File(exists=True, desc='resulting eroded mask')


class TPM2ROI(SimpleInterface):
    """
    Convert tissue probability maps (TPMs) into ROIs
    """

    input_spec = TPM2ROIInputSpec
    output_spec = TPM2ROIOutputSpec

    def _run_interface(self, runtime):
        roi_file, eroded_mask = _tpm2roi(
            self.inputs.in_file,
            self.inputs.in_mask,
            self.inputs.mask_erode_mm,
            self.inputs.erode_mm,
            self.inputs.prob_thresh
        )
        self._results['roi_file'] = roi_file
        self._results['eroded_mask'] = eroded_mask
        return runtime


class CombineROIsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='input list of ROIs')
    ref_header = File(exists=True, mandatory=True, desc='input mask')


class CombineROIsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output average file')


class CombineROIs(SimpleInterface):
    input_spec = CombineROIsInputSpec
    output_spec = CombineROIsOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _combine_rois(self.inputs.in_files, self.inputs.ref_header)
        return runtime


class ConcatROIsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    in_mask = File(exists=True, mandatory=True, desc='input file')
    ref_header = File(exists=True, mandatory=True, desc='input mask')


class ConcatROIsOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output average file')


class ConcatROIs(SimpleInterface):
    input_spec = ConcatROIsInputSpec
    output_spec = ConcatROIsOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _concat_rois(
            self.inputs.in_file, self.inputs.in_mask, self.inputs.ref_header)
        return runtime


class AddTSVHeaderInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input file')
    columns = traits.List(traits.Str, mandatory=True, desc='header for columns')


class AddTSVHeaderOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='output average file')


class AddTSVHeader(SimpleInterface):
    input_spec = AddTSVHeaderInputSpec
    output_spec = AddTSVHeaderOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _add_tsv_header(self.inputs.in_file, self.inputs.columns)
        return runtime


def _tpm2roi(in_file, epi_mask, epi_mask_erosion_mm=0, erosion_mm=0, pthres=0.95):
    """
    Generate a mask from a tissue probability map
    """
    probability_map_nii = resample_to_img(in_file, epi_mask)
    probability_map_data = probability_map_nii.get_data()

    # thresholding
    probability_map_data[probability_map_data < pthres] = 0
    probability_map_data[probability_map_data != 0] = 1

    epi_mask_nii = nb.load(epi_mask)
    epi_mask_data = epi_mask_nii.get_data()
    if epi_mask_erosion_mm > 0:
        epi_mask_data = nd.binary_erosion(
            epi_mask_data,
            iterations=int(epi_mask_erosion_mm /
                           max(probability_map_nii.header.get_zooms()))).astype(int)
        eroded_mask_file = os.path.abspath("erodd_mask.nii.gz")
        nb.Nifti1Image(epi_mask_data, epi_mask_nii.affine,
                       epi_mask_nii.header).to_filename(eroded_mask_file)
    else:
        eroded_mask_file = epi_mask
    probability_map_data[epi_mask_data != 1] = 0

    # shrinking
    if erosion_mm:
        iter_n = int(erosion_mm / max(probability_map_nii.header.get_zooms()))
        probability_map_data = nd.binary_erosion(probability_map_data,
                                                 iterations=iter_n).astype(int)

    new_nii = nb.Nifti1Image(probability_map_data, probability_map_nii.affine,
                             probability_map_nii.header)
    new_nii.to_filename("roi.nii.gz")
    return os.path.abspath("roi.nii.gz"), eroded_mask_file


def _combine_rois(in_files, ref_header):
    if len(in_files) < 2:
        raise RuntimeError('Combining ROIs requires at least two inputs')

    nii = nb.concat_images([nb.load(f) for f in in_files])
    combined = nii.get_data().astype(int).sum(3)
    combined[combined > 0] = 1
    # we have to do this explicitly because of potential differences in
    # qform_code between the two files that prevent aCompCor to work
    new_nii = nb.Nifti1Image(combined, nb.load(ref_header).affine,
                             nb.load(ref_header).header)
    new_nii.to_filename("logical_or.nii.gz")
    return os.path.abspath("logical_or.nii.gz")


def _concat_rois(in_file, in_mask, ref_header):
    nii = nb.load(in_file)
    mask_nii = nb.load(in_mask)

    # we have to do this explicitly because of potential differences in
    # qform_code between the two files that prevent SignalExtraction to do
    # the concatenation
    concat_nii = nb.concat_images([
        resample_to_img(nii, mask_nii, interpolation='nearest'), mask_nii])
    concat_nii = nb.Nifti1Image(concat_nii.get_data(),
                                nb.load(ref_header).affine,
                                nb.load(ref_header).header)
    concat_nii.to_filename("concat.nii.gz")
    return os.path.abspath("concat.nii.gz")


def _add_tsv_header(in_file, columns):
    out_file = fname_presuffix(in_file, suffix='_motion.tsv',
                               newpath=os.getcwd(),
                               use_ext=False)
    data = np.loadtxt(in_file)
    np.savetxt(out_file, data, delimiter='\t', header='\t'.join(columns),
               comments='')
    return out_file
