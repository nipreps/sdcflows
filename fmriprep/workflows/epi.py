#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
EPI MRI -processing workflows.

Originally coded by Craig Moodie. Refactored by the CRN Developers.
"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl

def sbref_workflow(name='SBrefPreprocessing', settings=None):

    if settings is None:
        settings = {}

    dwell_time = settings['epi'].get('dwell_time', 0.000700012460221792)


    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['sbref', 'mag_brain', 'fmap_scaled', 'fmap_mask', 'fmap_unmasked', 'in_topup']), 
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sbref_unwarped', 'sbref_fmap', 'mag2sbref_matrix']), name='outputnode')

    # Skull strip SBRef to get reference brain
    SBRef_BET = pe.Node(fsl.BET(mask=True, functional=True, frac=0.6), name="SBRef_BET")

    # Skull strip the SBRef with ANTS Brain Extraction

    #from nipype.interfaces.ants.segmentation import BrainExtraction
    #SBRef_skull_strip = pe.Node(BrainExtraction(), name = "antsreg_T1_Brain_Extraction")
    #SBRef_skull_strip.inputs.dimension = 3
    #SBRef_skull_strip.inputs.brain_template = "/home/cmoodie/Oasis_MICCAI2012-Multi-Atlas-Challenge-Data/T_template0.nii.gz"
    #SBRef_skull_strip.inputs.brain_probability_mask = "/home/cmoodie/Oasis_MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumProbabilityMask.nii.gz"
    #SBRef_skull_strip.inputs.extraction_registration_mask = "/home/cmoodie/Oasis_MICCAI2012-Multi-Atlas-Challenge-Data/T_template0_BrainCerebellumRegistrationMask.nii.gz"


    # Unwarp SBRef using Fugue  (N.B. duplicated in epi_reg_workflow!!!!!)
    fugue_sbref = pe.Node(fsl.FUGUE(unwarp_direction='x', dwell_time=dwell_time),
                          name="SBRef_Unwarping")

    strip_corrected_sbref = pe.Node(fsl.BET(mask=True, frac=0.6, robust=True),
                                    name="BET_Corrected_SBRef")

    ################################ Run the commands from epi_reg_dof #######
    # do a standard flirt pre-alignment
    # flirt -ref ${vrefbrain} -in ${vepi} -dof ${dof} -omat ${vout}_init.mat
    flt_epi_sbref = pe.Node(fsl.FLIRT(dof=6, bins=640, cost_func='mutualinfo'),
                            name="EPI_2_SBRef_Brain_Affine_Transform")

    # WITH FIELDMAP (unwarping steps)
    # flirt -in ${fmapmagbrain} -ref ${vrefbrain} -dof ${dof} -omat ${vout}_fieldmap2str_init.mat
    flt_fmap_mag_brain_sbref_brain = pe.Node(fsl.FLIRT(dof=6, bins=640, cost_func='mutualinfo'),
                                             name="Fmap_Mag_Brain_2_SBRef_Brain_Affine_Transform")

    # flirt -in ${fmapmaghead} -ref ${vrefhead} -dof ${dof} -init ${vout}_fieldmap2str_init.mat -omat ${vout}_fieldmap2str.mat -out ${vout}_fieldmap2str -nosearch
    flt_fmap_mag_sbref = pe.Node(fsl.FLIRT(dof=6, no_search=True, bins=640, cost_func='mutualinfo'),
                                 name="Fmap_Mag_2_SBRef_Affine_Transform")


    # the following is a NEW HACK to fix extrapolation when fieldmap is too small
    # applywarp -i ${vout}_fieldmaprads_unmasked -r ${vrefhead} --premat=${vout}_fieldmap2str.mat -o ${vout}_fieldmaprads2str_pad0
    aw_fmap_unmasked_sbref = pe.Node(
        fsl.ApplyWarp(relwarp=True), name="Apply_Warp_Fmap_Unmasked_2_SBRef")

    # fslmaths ${vout}_fieldmaprads2str_pad0 -abs -bin ${vout}_fieldmaprads2str_innermask
    fmap_unmasked_abs = pe.Node(
        fsl.UnaryMaths(operation='abs'), name="Abs_Fmap_Unmasked_Warp")
    fmap_unmasked_bin = pe.Node(
        fsl.UnaryMaths(operation='bin'), name="Binarize_Fmap_Unmasked_Warp")

    # fugue --loadfmap=${vout}_fieldmaprads2str_pad0 --mask=${vout}_fieldmaprads2str_innermask --unmaskfmap --unwarpdir=${fdir} --savefmap=${vout}_fieldmaprads2str_dilated
    fugue_dilate = pe.Node(fsl.FUGUE(unwarp_direction='x', dwell_time=dwell_time,
                                 save_unmasked_fmap=True), name="Fmap_Dilating")


    # Affine transform of T1 segmentation into SBRref space
    flt_wmseg_sbref = pe.Node(fsl.FLIRT(dof=6, bins=640, cost_func='mutualinfo'),
                              name="WMSeg_2_SBRef_Brain_Affine_Transform")


    workflow.connect([
        (inputnode, SBRef_BET, [('sbref', 'in_file')]),
        (inputnode, flt_wmseg_sbref, [('sbref', 'reference')]),
        (inputnode, fugue_sbref, [('sbref', 'in_file')]),
        (inputnode, fugue_sbref, [('fmap_scaled', 'fmap_in_file')]),
        (inputnode, flt_fmap_mag_brain_sbref_brain, [('mag_brain', 'in_file')]),
        (inputnode, fugue_sbref, [('fmap_mask', 'mask_file')]),
        (inputnode, flt_fmap_mag_sbref, [('in_topup', 'in_file')]),        
        (inputnode, aw_fmap_unmasked_sbref, [('fmap_unmasked', 'in_file')]),
        (fugue_sbref, strip_corrected_sbref, [('unwarped_file', 'in_file')]),
        # might need to switch to [strip_corrected_sbref, "in_file"] here
        # instead of [SBRef_BET, "out_file"]
        (SBRef_BET, flt_epi_sbref, [('out_file', 'reference')]),
        # might need to switch to [strip_corrected_sbref, "in_file"] here
        # instead of [SBRef_BET, "out_file"]
        (SBRef_BET, flt_fmap_mag_brain_sbref_brain, [('out_file', 'reference')]),
        (fugue_sbref, flt_fmap_mag_sbref, [('unwarped_file', 'reference')]),
        (flt_fmap_mag_brain_sbref_brain, flt_fmap_mag_sbref, [
            ('out_matrix_file', 'in_matrix_file')]),

        (flt_fmap_mag_sbref, aw_fmap_unmasked_sbref, [('out_matrix_file', 'premat')]),
        (fugue_sbref, aw_fmap_unmasked_sbref, [('unwarped_file', 'ref_file')]),
        (aw_fmap_unmasked_sbref, fmap_unmasked_abs, [('out_file', 'in_file')]),
        (fmap_unmasked_abs, fmap_unmasked_bin, [('out_file', 'in_file')]),
        (aw_fmap_unmasked_sbref, fugue_dilate, [('out_file', 'fmap_in_file')]),
        (fmap_unmasked_bin, fugue_dilate, [('out_file', 'mask_file')]),

        (fugue_sbref, outputnode, [('unwarped_file', 'sbref_unwarped')]),
        (fugue_dilate, outputnode, [('fmap_out_file', 'sbref_fmap')]),
        (flt_fmap_mag_sbref, outputnode, [('out_matrix_file', 'mag2sbref_matrix')]),

    ])
    return workflow
