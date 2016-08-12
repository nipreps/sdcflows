#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Apply susceptibility distortion correction (SDC)

"""
import os.path as op
from nipype.interfaces import fsl
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import ants
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from nipype.pipeline import engine as pe

from fmriprep.utils.misc import gen_list
from fmriprep.interfaces import ReadSidecarJSON
from fmriprep.workflows.fieldmap.base import create_encoding_file

SDC_CORRECT_NAME = 'SDC_Apply'

def sdc_correct(name=SDC_CORRECT_NAME, ref_vol=None):
    """
    This workflow takes an estimated fieldmap and a target image and applies TOPUP,
    an :abbr:`SDC (susceptibility-derived distortion correction)` method in FSL to
    unwarp the target image.

    Input fields:
    ~~~~~~~~~~~~~

      inputnode.in_file - the image(s) to which this correction will be applied
      inputnode.in_mask - a brain mask corresponding to the in_file image
      inputnode.fmap_ref - the fieldmap reference (generally, a *magnitude* image or the
                           resulting SE image)
      inputnode.fmap_mask - a brain mask in fieldmap-space
      inputnode.fieldmap - a fieldmap in Hz
      inputnode.hmc_movpar - the head motion parameters (iff inputnode.in_file is only
                             one 4D file)

    Output fields:
    ~~~~~~~~~~~~~~

      outputnode.out_file - the in_file after susceptibility-distortion correction.

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'in_mask', 'fmap_ref', 'fmap_mask', 'fieldmap',
                'hmc_movpar']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    # Compute movpar file iff we have several images with different
    # PE directions.
    align = pe.Node(niu.Function(
        input_names=['in_files', 'in_movpar', 'in_ref'],
        output_names=['out_file', 'out_movpar', 'method', 'ref_vol'],
        function=_multiple_pe_hmc), name='AlignMultiplePE')
    align.inputs.in_ref = ref_vol

    # Read metadata
    meta = pe.MapNode(ReadSidecarJSON(fields=['TotalReadoutTime', 'PhaseEncodingDirection']),
                      iterfield=['in_file'], name='metadata')

    encfile = pe.Node(interface=niu.Function(
        input_names=['input_images', 'in_dict'], output_names=['parameters_file'],
        function=create_encoding_file), name='TopUp_encfile', updatehash=True)

    fslsplit = pe.Node(fsl.Split(dimension='t'), name='ImageHMCSplit')

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    fmap2ref = pe.Node(ants.Registration(output_warped_image=True),
                       name='Fieldmap2ImageRegistration')
    fmap2ref.inputs.transforms = ['Translation', 'Rigid']
    fmap2ref.inputs.transform_parameters = [(0.5,), (.1,)]
    fmap2ref.inputs.number_of_iterations = [[50], [20]]
    fmap2ref.inputs.dimension = 3
    fmap2ref.inputs.metric = ['Mattes', 'Mattes']
    fmap2ref.inputs.metric_weight = [1.0] * 2
    fmap2ref.inputs.radius_or_number_of_bins = [64, 64]
    fmap2ref.inputs.sampling_strategy = ['Random', 'Random']
    fmap2ref.inputs.sampling_percentage = [0.2, 0.2]
    fmap2ref.inputs.convergence_threshold = [1.e-7, 1.e-8]
    fmap2ref.inputs.convergence_window_size = [20, 10]
    fmap2ref.inputs.smoothing_sigmas = [[10.0], [2.0]]
    fmap2ref.inputs.sigma_units = ['mm'] * 2
    fmap2ref.inputs.shrink_factors = [[4], [1]]  # ,[1] ]
    fmap2ref.inputs.use_estimate_learning_rate_once = [True] * 2
    fmap2ref.inputs.use_histogram_matching = [True] * 2
    fmap2ref.inputs.initial_moving_transform_com = 0
    fmap2ref.inputs.collapse_output_transforms = True
    fmap2ref.inputs.winsorize_upper_quantile = 0.998
    fmap2ref.inputs.winsorize_lower_quantile = 0.015

    applyxfm = pe.Node(ants.ApplyTransforms(
        dimension=3, interpolation='Linear'), name='Fieldmap2ImageApply')

    add_dims = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'], function=_add_empty), name='Fmap4D')

    # Convert to valid fieldcoeff:
    # 1. Warputils; 2. Remove dims and set correct header.
    to_coeff = pe.Node(fsl.WarpUtils(out_format='spline', warp_resolution=(4., 4., 4.)),
                       name='ComputeCoeffs')


    # Use the least-squares method to correct the dropout of the SBRef images
    unwarp = pe.Node(fsl.ApplyTOPUP(), name='TopUpApply')


    if ref_vol is None:
        getref = pe.Node(fsl.MeanImage(dimension='T'), name='ImageAverage')

        workflow.connect([
            (align, getref, [('out_file', 'in_file')]),
            (getref, fmap2ref, [('out_file', 'fixed_image')]),
            (getref, applyxfm, [('out_file', 'reference_image')]),
            (getref, to_coeff, [('out_file', 'reference')]),
        ])
    else:
        getref = pe.Node(niu.Select(), name='PickFirst')
        workflow.connect([
            (fslsplit, getref, [('out_files', 'inlist')]),
            (align, getref, [('ref_vol', 'index')]),
            (getref, fmap2ref, [('out', 'fixed_image')]),
            (getref, applyxfm, [('out', 'reference_image')]),
            (getref, to_coeff, [('out', 'reference')]),
        ])


    workflow.connect([
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, align, [('in_file', 'in_files'),
                            ('hmc_movpar', 'in_movpar')]),
        (inputnode, applyxfm, [('fieldmap', 'input_image')]),
        (inputnode, encfile, [('in_file', 'input_images')]),
        (meta, encfile, [('out_dict', 'in_dict')]),

        (align, unwarp, [('out_movpar', 'in_topup_movpar'),
                         ('method', 'method')]),

        (inputnode, fmap2ref, [('in_mask', 'fixed_image_mask'),
                               ('fmap_ref', 'moving_image'),
                               ('fmap_mask', 'moving_image_mask')]),
        (fmap2ref, applyxfm, [
            ('forward_transforms', 'transforms'),
            ('forward_invert_flags', 'invert_transform_flags')]),
        (align, fslsplit, [('out_file', 'in_file')]),
        (applyxfm, add_dims, [('output_image', 'in_file')]),
        (add_dims, to_coeff, [('out_file', 'in_file')]),
        (fslsplit, unwarp, [('out_files', 'in_files'),
                            (('out_files', gen_list), 'in_index')]),
        (to_coeff, unwarp, [('out_file', 'in_topup_fieldcoef')]),
        (encfile, unwarp, [('parameters_file', 'encoding_file')]),
        (unwarp, outputnode, [('out_corrected', 'out_file')])
    ])
    return workflow


def _multiple_pe_hmc(in_files, in_movpar, in_ref):
    """
    This function interprets that we are dealing with a
    multiple PE (phase encoding) input if it finds several
    files in in_files.

    If we have several images with various PE directions,
    it will compute the HMC parameters between them using
    an embedded workflow.

    It just forwards the two inputs otherwise.
    """
    if len(in_files) == 1:
        return in_files[0], in_movpar, 'jac', in_ref
    else:
        import os
        from nipype.interfaces import fsl
        from fmripep.interfaces import FormatHMCParam

        if in_ref is None:
            in_ref = 0

        # Head motion correction
        fslmerge = fsl.Merge(dimension='t')
        fslmerge.inputs.in_file = in_files
        hmc = fsl.MCFLIRT(cost='normcorr', ref_vol=in_ref, save_mats=True)
        hmc.inputs.in_file = fslmerge.run().outputs.merged_file
        hmc_res = hmc.run()

        translations = []
        rot_angles = []
        for mat_file in hmc_res.outputs.mat_file:
            avscale = fsl.utils.AvScale(
                all_param=True, mat_file=mat_file, ref_file=in_files[0])
            avres = avscale.run()
            translations.append(avres.outputs.translations)
            rot_angles.append(avres.outputs.rot_angles)

        avs_format = FormatHMCParam(translations=translations,
                                    rot_angles=rot_angles,
                                    fmt='movpar_file')
        return (hmc_res.outputs.out_file,
                avs_format.run().outputs.out_file,
                'lsr', in_ref)


def _add_empty(in_file, out_file=None):
    import os.path as op
    import numpy as np
    import nibabel as nb

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_fmap4D.nii.gz' % fname)

    im0 = nb.load(in_file)
    data = np.zeros_like(im0.get_data())
    im1 = nb.Nifti1Image(data, im0.get_affine(), im0.get_header())
    im4d = nb.concat_images([im0, im1, im1])
    im4d.to_filename(out_file)
    return out_file
