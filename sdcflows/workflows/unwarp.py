#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Apply susceptibility distortion correction (SDC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. topic :: Abbreviations

    fmap
        fieldmap
    VSM
        voxel-shift map -- a 3D nifti where displacements are in pixels (not mm)
    DFM
        displacements field map -- a nifti warp file compatible with ANTs (mm)

"""
from __future__ import print_function, division, absolute_import, unicode_literals

import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces.fsl import FUGUE
from nipype.interfaces.ants import (Registration, N4BiasFieldCorrection,
                                    CreateJacobianDeterminantImage)
# from nipype.interfaces.ants.preprocess import Matrix2FSLParams
from niworkflows.interfaces.registration import ANTSApplyTransformsRPT

from fmriprep.interfaces import itk
from fmriprep.interfaces.epi import SelectReference
from fmriprep.interfaces.nilearn import Mean, MaskEPI
from fmriprep.interfaces import ReadSidecarJSON


def sdc_unwarp(name='SDC_unwarp', settings=None):
    """
    This workflow takes in a displacements fieldmap and calculates the corresponding
    displacements field (in other words, an ANTs-compatible warp file).

    .. workflow ::

        from fmriprep.workflows.fieldmap.unwarp import sdc_unwarp
        wf = sdc_unwarp()


    Inputs

        in_split
            the input image to be corrected split in 3D files
        in_reference
            the reference image (generally, the average of ``in_split``)
        in_mask
            a brain mask corresponding to ``in_split`` and ``in_reference``
        in_meta
            a dictionary of metadata corresponding to ``in_split``
        fmap
            the fieldmap in Hz
        fmap_ref
            the reference (anatomical) image corresponding to ``fmap``
        fmap_mask
            a brain mask corresponding to ``fmap``


    Outputs

        out_files
            the ``in_split`` files after unwarping
        out_reference
            the ``in_reference`` or the mean ``in_split`` after unwarping
        out_warp
            the corresponding :abbr:`DFM (displacements field map)` compatible with
            ANTs
        out_jacobian
            the jacobian of the field (for drop-out alleviation)

    """
    from fmriprep.interfaces.fmap import WarpReference
    from fmriprep.interfaces.utils import ApplyMask

    if settings is None:
        # Don't crash if workflow used outside fmriprep
        settings = {'ants_nthreads': 6}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_split', 'in_reference', 'in_mask', 'xforms', 'name_source',
                'fmap_ref', 'fmap_mask', 'fmap']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_files', 'out_reference', 'out_warps', 'out_mask',
                'out_jacobian']), name='outputnode')

    meta = pe.Node(ReadSidecarJSON(), name='metadata')

    ref_img = pe.Node(SelectReference(), name='ref_select')

    # Prepare target image for registration
    ref_inu = pe.Node(N4BiasFieldCorrection(dimension=3), name='ref_inu')

    ants_init = pe.Node(itk.AffineInitializer(), name='ants_init')
    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    ants_settings = pkgr.resource_filename('fmriprep', 'data/fmap-any_registration.json')
    if settings.get('debug', False):
        ants_settings = pkgr.resource_filename(
            'fmriprep', 'data/fmap-any_registration_testing.json')
    fmap2ref_reg = pe.Node(Registration(
        from_file=ants_settings, output_inverse_warped_image=True,
        output_warped_image=True, num_threads=settings['ants_nthreads']),
                       name='fmapref2ref0')
    fmap2ref_reg.interface.num_threads = settings['ants_nthreads']

    # Map the VSM into the EPI space
    fmap2ref_apply = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='BSpline', float=True),
                             name='fmap2ref')

    # Map the mask
    fmapmask2ref = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='NearestNeighbor', float=True),
                             name='fmap_mask2ref')


    # Fieldmap to rads and then to voxels (VSM - voxel shift map)
    torads = pe.Node(niu.Function(input_names=['in_file'], output_names=['out_file'],
                                  function=_hz2rads), name='fmap_hz2rads')



    gen_vsm = pe.Node(FUGUE(save_unmasked_shift=True), name='fmap_shiftmap')
    # Convert the VSM into a DFM (displacements field map)
    # or: FUGUE shift to ANTS warping.
    vsm2dfm = pe.Node(itk.FUGUEvsm2ANTSwarp(), name='fmap2dfm')
    jac_dfm = pe.Node(CreateJacobianDeterminantImage(
        imageDimension=3, outputImage='jacobian.nii.gz'), name='jacobian_det')

    unwarp = pe.MapNode(ANTSApplyTransformsRPT(
        dimension=3, generate_report=False, float=True, interpolation='LanczosWindowedSinc'),
                        iterfield=['input_image'], name='unwarp_all')
    ref_avg = pe.Node(Mean(), name='mean')
    ref_msk_post = pe.Node(MaskEPI(), name='mask_post')
    ref_avg_inu = pe.Node(N4BiasFieldCorrection(dimension=3), name='ref_avg_inu')

    # Final correction with refined HMC parameters
    tfm_concat = pe.MapNode(itk.MergeANTsTransforms(
        in_file_invert=False, invert_transform_flags=[False]),
                            iterfield=['in_file'], name='concat_hmc_sdc_xforms')

    msk_combine = pe.Node(niu.Function(
        input_names=['ref_msk', 'corrected_msk'], output_names=['out_file'],
        function=_mskcomb), name='mask_combine')

    workflow.connect([
        (inputnode, meta, [('name_source', 'in_file')]),
        (inputnode, ref_img, [('in_reference', 'reference'),
                              ('in_split', 'in_files')]),
        (inputnode, tfm_concat, [('xforms', 'in_file')]),
        (inputnode, fmap2ref_reg, [('fmap_ref', 'fixed_image')]),
        (inputnode, ants_init, [('fmap_ref', 'fixed_image')]),
        (inputnode, fmap2ref_apply, [('fmap', 'input_image')]),
        (ref_img, fmap2ref_apply, [('reference', 'reference_image')]),
        (fmap2ref_reg, fmap2ref_apply, [
            ('inverse_composite_transform', 'transforms')]),
        (ref_img, ref_inu, [('reference', 'input_image')]),
        (ref_inu, ants_init, [('output_image', 'moving_image')]),
        (ants_init, fmap2ref_reg, [('out_file', 'initial_moving_transform')]),
        (ref_inu, fmap2ref_reg, [('output_image', 'moving_image')]),
        (inputnode, fmapmask2ref, [('fmap_mask', 'input_image')]),
        (ref_img, fmapmask2ref, [('reference', 'reference_image')]),
        (fmap2ref_reg, fmapmask2ref, [
            ('inverse_composite_transform', 'transforms')]),
        (fmap2ref_apply, torads, [('output_image', 'in_file')]),
        (meta, gen_vsm, [(('out_dict', _get_ec), 'dwell_time'),
                         (('out_dict', _get_pedir), 'unwarp_direction')]),
        (meta, vsm2dfm, [(('out_dict', _get_pedir), 'pe_dir')]),
        (torads, gen_vsm, [('out_file', 'fmap_in_file')]),
        (vsm2dfm, unwarp, [('out_file', 'transforms')]),
        (ref_img, unwarp, [('reference', 'reference_image')]),
        (inputnode, unwarp, [('in_split', 'input_image')]),
        (unwarp, ref_avg, [('output_image', 'in_files')]),
        (vsm2dfm, tfm_concat, [('out_file', 'transforms')]),
        (vsm2dfm, jac_dfm, [('out_file', 'deformationField')]),
        (ref_avg, ref_msk_post, [('out_file', 'in_files')]),
        (ref_avg, ref_avg_inu, [('out_file', 'input_image')]),
        (ref_avg_inu, outputnode, [('output_image', 'out_reference')]),
        (unwarp, outputnode, [('output_image', 'out_files')]),
        (tfm_concat, outputnode, [('transforms', 'out_warps')]),
        (ref_msk_post, msk_combine, [('out_mask', 'corrected_msk')]),
        (fmapmask2ref, msk_combine, [('output_image', 'ref_msk')]),
        (msk_combine, outputnode, [('out_file', 'out_mask')]),
        (jac_dfm, outputnode, [('jacobian_image', 'out_jacobian')]),
    ])

    if not settings.get('fmap_bspline', False):
        workflow.connect([
            (fmapmask2ref, gen_vsm, [('output_image', 'mask_file')])
        ])

    if settings.get('fmap-demean', True):
        # Demean within mask
        demean = pe.Node(niu.Function(
            input_names=['in_file', 'in_mask'], output_names=['out_file'],
            function=_demean), name='fmap_demean')

        workflow.connect([
            (gen_vsm, demean, [('shift_out_file', 'in_file')]),
            (fmapmask2ref, demean, [('output_image', 'in_mask')]),
            (demean, vsm2dfm, [('out_file', 'in_file')]),
        ])

    else:
        workflow.connect([
            (gen_vsm, vsm2dfm, [('shift_out_file', 'in_file')]),
        ])

    return workflow

# Helper functions
# ------------------------------------------------------------

def _get_ec(in_dict):
    return float(in_dict['EffectiveEchoSpacing'])

def _get_pedir(in_dict):
    return in_dict['PhaseEncodingDirection'].replace('j', 'y').replace('i', 'x')

def _hz2rads(in_file, out_file=None):
    """Transform a fieldmap in Hz into rad/s"""
    from math import pi
    import nibabel as nb
    from fmriprep.utils.misc import genfname
    if out_file is None:
        out_file = genfname(in_file, 'rads')
    nii = nb.load(in_file)
    data = nii.get_data() * 2.0 * pi
    nb.Nifti1Image(data, nii.get_affine(),
                   nii.get_header()).to_filename(out_file)
    return out_file

def _demean(in_file, in_mask, out_file=None):
    import numpy as np
    import nibabel as nb
    from fmriprep.utils.misc import genfname

    if out_file is None:
        out_file = genfname(in_file, 'demeaned')
    nii = nb.load(in_file)
    msk = nb.load(in_mask).get_data()
    data = nii.get_data()
    data -= np.median(data[msk > 0])
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(
        out_file)
    return out_file

def _mskcomb(ref_msk, corrected_msk, out_file=None):
    import nibabel as nb
    import numpy as np
    from fmriprep.utils.misc import genfname

    if out_file is None:
        out_file = genfname(corrected_msk, 'mask')

    nii = nb.load(corrected_msk)
    data = nii.get_data().astype(np.uint8)
    data += nb.load(ref_msk).get_data().astype(np.uint8)
    data[data > 0] = 1
    hdr = nii.header.copy()
    hdr.set_data_dtype(np.uint8)
    nb.Nifti1Image(data.astype(np.uint8), nii.affine, hdr).to_filename(
        out_file)
    return out_file
