#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Apply susceptibility distortion correction (SDC)

"""
import os.path as op
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from nipype.pipeline import engine as pe

from fmriprep.utils.misc import gen_list
from fmriprep.interfaces import ReadSidecarJSON
from fmriprep.workflows.fieldmap.base import create_encoding_file

SDC_CORRECT_NAME = 'SDC_Apply'

def sdc_correct(name=SDC_CORRECT_NAME, topup_meth='lsr', bet_frac=0.6):
    """
    This workflow takes an estimated fieldmap and a target image and applies TOPUP,
    an :abbr:`SDC (susceptibility-derived distortion correction)` method in FSL to
    unwarp the target image.

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'fmap_ref', 'fmap_mask', 'fieldmap',
                'hmc_movpar']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    # Read metadata
    meta = pe.MapNode(ReadSidecarJSON(fields=['TotalReadoutTime', 'PhaseEncodingDirection']),
                      iterfield=['in_file'], name='metadata')

    encfile = pe.Node(interface=niu.Function(
        input_names=['input_images', 'in_dict'], output_names=['parameters_file'],
        function=create_encoding_file), name='TopUp_encfile', updatehash=True)

    # Head motion correction
    fslmerge = pe.Node(fsl.Merge(dimension='t'), name='ImageMerge')
    hmc_se = pe.Node(fsl.MCFLIRT(
        cost='normcorr', ref_vol=0, save_mats=True), name='ImageHMC')
    fslsplit = pe.Node(fsl.Split(dimension='t'), name='ImageHMCSplit')

    # Remove bias
    inu_n4 = pe.Node(N4BiasFieldCorrection(dimension=3), name='SBref_bias')

    # Skull-strip target image
    bet_target = pe.Node(fsl.BET(mask=True, functional=True, frac=bet_frac),
                         name="ImageBET")

    fmap2ref = pe.Node(ants.Registration(output_warped_image=True),
                       name='Fieldmap2ImageRegistration')
    fmap2ref.inputs.transforms = ['Rigid'] * 2
    fmap2ref.inputs.transform_parameters = [(1.0,)] * 2
    fmap2ref.inputs.number_of_iterations = [[50], [20]]
    fmap2ref.inputs.dimension = 3
    fmap2ref.inputs.metric = ['Mattes', 'Mattes']
    fmap2ref.inputs.metric_weight = [1.0] * 2
    fmap2ref.inputs.radius_or_number_of_bins = [64, 64]
    fmap2ref.inputs.sampling_strategy = ['Regular', 'Random']
    fmap2ref.inputs.sampling_percentage = [None, 0.2]
    fmap2ref.inputs.convergence_threshold = [1.e-5, 1.e-8]
    fmap2ref.inputs.convergence_window_size = [20, 10]
    fmap2ref.inputs.smoothing_sigmas = [[6.0], [2.0]]
    fmap2ref.inputs.sigma_units = ['vox'] * 2
    fmap2ref.inputs.shrink_factors = [[6], [1]]  # ,[1] ]
    fmap2ref.inputs.use_estimate_learning_rate_once = [True] * 2
    fmap2ref.inputs.use_histogram_matching = [True] * 2
    fmap2ref.inputs.initial_moving_transform_com = 0
    fmap2ref.inputs.collapse_output_transforms = True
    fmap2ref.inputs.winsorize_upper_quantile = 0.995

    applyxfm = pe.Node(ants.ApplyTransforms(
        dimension=3, interpolation='Linear'), name='Fieldmap2ImageApply')

    # Use the least-squares method to correct the dropout of the SBRef images
    unwarp = pe.Node(fsl.ApplyTOPUP(method=topup_meth), name='TopUpApply')

    to_coeff = pe.Node(fsl.WarpUtils(out_format='spline'), name='ComputeCoeffs')

    workflow.connect([
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, fslmerge, [('in_file', 'in_files')]),
        (inputnode, applyxfm, [('fieldmap', 'input_image')]),
        (inputnode, unwarp, [('hmc_movpar', 'in_topup_movpar')]),
        (inputnode, encfile, [('in_file', 'input_images')]),
        (meta, encfile, [('out_dict', 'in_dict')]),
        (fslmerge, hmc_se, [('merged_file', 'in_file')]),
        (hmc_se, inu_n4, [('out_file', 'input_image')]),
        (inu_n4, bet_target, [('output_image', 'in_file')]),
        (bet_target, fmap2ref, [('out_file', 'fixed_image'),
                                ('mask_file', 'fixed_image_mask')]),
        (inputnode, fmap2ref, [('fmap_ref', 'moving_image'),
                               ('fmap_mask', 'moving_image_mask')]),
        (fmap2ref, applyxfm, [
            ('forward_transforms', 'transforms'),
            ('forward_invert_flags', 'invert_transform_flags')]),

        (bet_target, applyxfm, [('out_file', 'reference_image')]),
        (applyxfm, to_coeff, [('output_image', 'in_file')]),
        (bet_target, to_coeff, [('out_file', 'reference')]),
        (hmc_se, fslsplit, [('out_file', 'in_file')]),
        (fslsplit, unwarp, [('out_files', 'in_files'),
                            (('out_files', gen_list), 'in_index')]),
        (to_coeff, unwarp, [('out_file', 'in_topup_fieldcoef')]),
        (encfile, unwarp, [('parameters_file', 'encoding_file')]),
        (unwarp, outputnode, [('out_corrected', 'out_file')])
    ])
    return workflow
