#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to deal with the various types of fieldmap sources

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os.path as op
from shutil import copy
from builtins import range
import numpy as np
import nibabel as nb
from nipype import logging
from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
                                    File, isdefined, traits, InputMultiPath, Str)
from nipype.interfaces import fsl
from niworkflows.interfaces.registration import FUGUERPT
from fmriprep.utils.misc import genfname

LOGGER = logging.getLogger('interfaces')


class WarpReferenceInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    in_mask = File(exists=True, mandatory=True, desc='mask of fieldmap')
    fmap_ref = File(exists=True, mandatory=True, desc='reference file')
    echospacing = traits.Float(mandatory=True,
                               desc='effective echo spacing or dwell time')
    pe_dir = traits.Enum('y', 'x', 'y-', 'x-', mandatory=True,
                         desc='phase encoding direction')

class WarpReferenceOutputSpec(TraitedSpec):
    out_warped = File(desc='the "fmap_ref" image after warping')
    out_mask = File(desc='the corresponding fieldmap mask warped')

class WarpReference(BaseInterface):
    """
    The WarpReference interface wraps a workflow to generate
    a warped version of the reference (generally magnitude) image.
    This way, the reference image is geometrically closer to the
    target EPI and thus, easier to register.
    """
    input_spec = WarpReferenceInputSpec
    output_spec = WarpReferenceOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(WarpReference, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        eec = self.inputs.echospacing
        ped = self.inputs.pe_dir

        vsm_file = genfname(self.inputs.fmap_ref, 'vsm')
        gen_vsm = fsl.FUGUE(
            fmap_in_file=self.inputs.in_file, dwell_time=eec,
            unwarp_direction=ped, shift_out_file=vsm_file)
        gen_vsm.run()

        fugue = fsl.FUGUE(
            in_file=self.inputs.fmap_ref, shift_in_file=vsm_file, nokspace=True,
            forward_warping=True, unwarp_direction=ped, icorr=False,
            warped_file=genfname(self.inputs.fmap_ref, 'warped%s' % ped)).run()

        fuguemask = fsl.FUGUE(
            in_file=self.inputs.in_mask, shift_in_file=vsm_file, nokspace=True,
            forward_warping=True, unwarp_direction=ped, icorr=False,
            warped_file=genfname(self.inputs.in_mask, 'maskwarped%s' % ped)).run()

        masknii = nb.load(fuguemask.outputs.warped_file)
        mask = masknii.get_data()
        mask[mask > 0] = 1
        mask[mask < 1] = 0
        out_mask = genfname(self.inputs.in_mask, 'warp_bin')
        nb.Nifti1Image(mask.astype(np.uint8), masknii.affine,
                       masknii.header).to_filename(out_mask)

        self._results['out_warped'] = fugue.outputs.warped_file
        self._results['out_mask'] = out_mask

        return runtime


class FieldEnhanceInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    in_mask = File(exists=True, desc='brain mask')
    despike = traits.Bool(True, usedefault=True, desc='run despike filter')
    bspline_smooth = traits.Bool(True, usedefault=True, desc='run 3D bspline smoother')
    mask_erode = traits.Int(1, usedefault=True, desc='mask erosion iterations')
    despike_threshold = traits.Float(0.2, usedefault=True, desc='mask erosion iterations')
    njobs = traits.Int(1, usedefault=True, nohash=True, desc='number of jobs')

class FieldEnhanceOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')
    out_coeff = File(desc='write bspline coefficients')

class FieldEnhance(BaseInterface):
    """
    The FieldEnhance interface wraps a workflow to massage the input fieldmap
    and return it masked, despiked, etc.
    """
    input_spec = FieldEnhanceInputSpec
    output_spec = FieldEnhanceOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(FieldEnhance, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):
        from scipy import ndimage as sim

        fmap_nii = nb.load(self.inputs.in_file)
        data = np.squeeze(fmap_nii.get_data().astype(np.float32))

        # Despike / denoise (no-mask)
        if self.inputs.despike:
            data = _despike2d(data, self.inputs.despike_threshold)

        mask = None
        if isdefined(self.inputs.in_mask):
            masknii = nb.load(self.inputs.in_mask)
            mask = masknii.get_data().astype(np.uint8)

            # Dilate mask
            if self.inputs.mask_erode > 0:
                struc = sim.iterate_structure(sim.generate_binary_structure(3, 2), 1)
                mask = sim.binary_erosion(
                    mask, struc,
                    iterations=self.inputs.mask_erode).astype(np.uint8)  # pylint: disable=no-member

        self._results['out_file'] = genfname(self.inputs.in_file, suffix='enh')
        self._results['out_coeff'] = genfname(self.inputs.in_file, suffix='coeff')

        datanii = nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header)
        # data interpolation
        datanii.to_filename(self._results['out_file'])

        if self.inputs.bspline_smooth:
            from fmriprep.utils import bspline as fbsp
            from statsmodels.robust.scale import mad

            # Fit BSplines (coarse)
            bspobj = fbsp.BSplineFieldmap(datanii, weights=mask, njobs=self.inputs.njobs)
            bspobj.fit()
            smoothed1 = bspobj.get_smoothed()

            # Manipulate the difference map
            diffmap = data - smoothed1.get_data()
            sderror = mad(diffmap[mask > 0])
            errormask = np.zeros_like(diffmap)
            errormask[np.abs(diffmap) > (10 * sderror)] = 1
            errormask *= mask
            errorslice = np.squeeze(np.argwhere(errormask.sum(0).sum(0) > 0))

            if (errorslice[-1] - errorslice[0]) > 1:
                diffmapmsk = mask[..., errorslice[0]:errorslice[-1]]
                diffmapnii = nb.Nifti1Image(
                    diffmap[..., errorslice[0]:errorslice[-1]] * diffmapmsk,
                    datanii.affine, datanii.header)


                bspobj2 = fbsp.BSplineFieldmap(diffmapnii, knots_zooms=[24., 24., 4.],
                                               njobs=self.inputs.njobs)
                bspobj2.fit()
                smoothed2 = bspobj2.get_smoothed().get_data()

                final = smoothed1.get_data().copy()
                final[..., errorslice[0]:errorslice[-1]] += smoothed2
            else:
                final = smoothed1

            nb.Nifti1Image(final, datanii.affine, datanii.header).to_filename(
                self._results['out_file'])

        return runtime


def _despike2d(data, thres, neigh=None):
    """
    despiking as done in FSL fugue
    """

    if neigh is None:
        neigh = [-1, 0, 1]
    nslices = data.shape[-1]

    for k in range(nslices):
        data2d = data[..., k]

        for i in range(data2d.shape[0]):
            for j in range(data2d.shape[1]):
                vals = []
                thisval = data2d[i, j]
                for ii in neigh:
                    for jj in neigh:
                        try:
                            vals.append(data2d[i + ii, j + jj])
                        except IndexError:
                            pass
                vals = np.array(vals)
                patch_range = vals.max() - vals.min()
                patch_med = np.median(vals)

                if (patch_range > 1e-6 and
                        (abs(thisval - patch_med) / patch_range) > thres):
                    data[i, j, k] = patch_med
    return data
