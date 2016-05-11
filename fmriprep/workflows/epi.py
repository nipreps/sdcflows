#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
EPI MRI -processing workflows.

Originally coded by Craig Moodie. Refactored by the CRN Developers.
"""

import os
import os.path as op
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl


def sbref_workflow(name='SBrefPreprocessing', settings=None):
    """SBref processing workflow"""
    if settings is None:
        settings = {}

    dwell_time = settings['epi'].get('dwell_time', 0.000700012460221792)

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['sbref', 'mag_brain', 'fmap_scaled', 'fmap_mask', 'fmap_unmasked',
                'in_topup']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['sbref_unwarped', 'sbref_fmap', 'mag2sbref_matrix', 'sbref_brain',
                'sbref_brain_corrected', 't1_brain']), name='outputnode')

    # Skull strip SBRef to get reference brain
    sbref_bet = pe.Node(
        fsl.BET(mask=True, functional=True, frac=0.6), name="sbref_bet")

    # Skull strip the SBRef with ANTS Brain Extraction

    #from nipype.interfaces.ants.segmentation import BrainExtraction
    #SBRef_skull_strip = pe.Node(BrainExtraction(), name = "antsreg_T1_Brain_Extraction")
    #SBRef_skull_strip.inputs.dimension = 3
    #SBRef_skull_strip.inputs.brain_template = "T_template0.nii.gz"
    #SBRef_skull_strip.inputs.brain_probability_mask = "T_template0_BrainCerebellumProbabilityMask.nii.gz"
    #SBRef_skull_strip.inputs.extraction_registration_mask = "T_template0_BrainCerebellumRegistrationMask.nii.gz"

    # Unwarp SBRef using Fugue  (N.B. duplicated in epi_reg_workflow!!!!!)
    fugue_sbref = pe.Node(fsl.FUGUE(unwarp_direction='x', dwell_time=dwell_time),
                          name="SBRef_Unwarping")

    strip_corrected_sbref = pe.Node(fsl.BET(mask=True, frac=0.6, robust=True),
                                    name="BET_Corrected_SBRef")

    ################################ Run the commands from epi_reg_dof #######
    # WITH FIELDMAP (unwarping steps)
    # flirt -in ${fmapmagbrain} -ref ${vrefbrain} -dof ${dof} -omat
    # ${vout}_fieldmap2str_init.mat
    flt_fmap_mag_brain_sbref_brain = pe.Node(fsl.FLIRT(dof=6, bins=640, cost_func='mutualinfo'),
                                             name="Fmap_Mag_Brain_2_SBRef_Brain_Affine_Transform")

    # flirt -in ${fmapmaghead} -ref ${vrefhead} -dof ${dof} -init
    # ${vout}_fieldmap2str_init.mat -omat ${vout}_fieldmap2str.mat -out
    # ${vout}_fieldmap2str -nosearch
    flt_fmap_mag_sbref = pe.Node(fsl.FLIRT(dof=6, no_search=True, bins=640, cost_func='mutualinfo'),
                                 name="Fmap_Mag_2_SBRef_Affine_Transform")

    # the following is a NEW HACK to fix extrapolation when fieldmap is too small
    # applywarp -i ${vout}_fieldmaprads_unmasked -r ${vrefhead}
    # --premat=${vout}_fieldmap2str.mat -o ${vout}_fieldmaprads2str_pad0
    aw_fmap_unmasked_sbref = pe.Node(
        fsl.ApplyWarp(relwarp=True), name="Apply_Warp_Fmap_Unmasked_2_SBRef")

    # fslmaths ${vout}_fieldmaprads2str_pad0 -abs -bin
    # ${vout}_fieldmaprads2str_innermask
    fmap_unmasked_abs = pe.Node(
        fsl.UnaryMaths(operation='abs'), name="Abs_Fmap_Unmasked_Warp")
    fmap_unmasked_bin = pe.Node(
        fsl.UnaryMaths(operation='bin'), name="Binarize_Fmap_Unmasked_Warp")

    # fugue --loadfmap=${vout}_fieldmaprads2str_pad0
    # --mask=${vout}_fieldmaprads2str_innermask --unmaskfmap
    # --unwarpdir=${fdir} --savefmap=${vout}_fieldmaprads2str_dilated
    fugue_dilate = pe.Node(fsl.FUGUE(unwarp_direction='x', dwell_time=dwell_time,
                                     save_unmasked_fmap=True), name="Fmap_Dilating")

    # epi_reg --epi=sub-S2529LVY1263171_task-nback_run-1_bold_brain
    # --t1=../Preprocessing_test_workflow/_subject_id_S2529LVY1263171/Bias_Field_Correction/sub-S2529LVY1263171_run-1_T1w_corrected.nii.gz
    # --t1brain=sub-S2529LVY1263171_run-1_T1w_corrected_bet_brain.nii.gz
    # --out=sub-S2529LVY1263171/func/sub-S2529LVY1263171_task-nback_run-1_bold_undistorted
    # --fmap=Topup_Fieldmap_rad.nii.gz
    # --fmapmag=fieldmap_topup_corrected.nii.gz
    # --fmapmagbrain=Magnitude_brain.nii.gz --echospacing=dwell_time
    # --pedir=x- -v


    workflow.connect([
        (inputnode, sbref_bet, [('sbref', 'in_file')]),
        (inputnode, fugue_sbref, [('sbref', 'in_file')]),
        (inputnode, fugue_sbref, [('fmap_scaled', 'fmap_in_file')]),
        (inputnode, flt_fmap_mag_brain_sbref_brain, [('mag_brain', 'in_file')]),
        (inputnode, fugue_sbref, [('fmap_mask', 'mask_file')]),
        (inputnode, flt_fmap_mag_sbref, [('in_topup', 'in_file')]),
        (inputnode, aw_fmap_unmasked_sbref, [('fmap_unmasked', 'in_file')]),
        #(inputnode, SBRef_skull_strip, [("sbref", "in_file")]),
        (fugue_sbref, strip_corrected_sbref, [('unwarped_file', 'in_file')]),
        # might need to switch to [strip_corrected_sbref, "in_file"] here
        # instead of [sbref_bet, "out_file"]
        # might need to switch to [strip_corrected_sbref, "in_file"] here
        # instead of [sbref_bet, "out_file"]
        (sbref_bet, flt_fmap_mag_brain_sbref_brain, [('out_file', 'reference')]),
        (fugue_sbref, flt_fmap_mag_sbref, [('unwarped_file', 'reference')]),
        (flt_fmap_mag_brain_sbref_brain, flt_fmap_mag_sbref, [
            ('out_matrix_file', 'in_matrix_file')]),
        (flt_fmap_mag_sbref, aw_fmap_unmasked_sbref, [('out_matrix_file', 'premat')]),
        (fugue_sbref, aw_fmap_unmasked_sbref, [('unwarped_file', 'ref_file')]),
        (aw_fmap_unmasked_sbref, fmap_unmasked_abs, [('out_file', 'in_file')]),
        (fmap_unmasked_abs, fmap_unmasked_bin, [('out_file', 'in_file')]),
        (aw_fmap_unmasked_sbref, fugue_dilate, [('out_file', 'fmap_in_file')]),
        (fmap_unmasked_bin, fugue_dilate, [('out_file', 'mask_file')]),

        (sbref_bet, outputnode, [('out_file', 'sbref_brain')]),
        (fugue_sbref, outputnode, [('unwarped_file', 'sbref_unwarped')]),
        (fugue_dilate, outputnode, [('fmap_out_file', 'sbref_fmap')]),
        (flt_fmap_mag_sbref, outputnode, [('out_matrix_file', 'mag2sbref_matrix')]),
        (strip_corrected_sbref, outputnode, [('out_file', 'sbref_brain_corrected')])
    ])
    return workflow


def correction_workflow(name='EPIUnwarpWorkflow', settings=None):  # pylint: disable=R0914
    """ A workflow to correct EPI images """
    if settings is None:
        settings = {}

    dwell_time = settings['epi'].get('dwell_time', 0.000700012460221792)

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['epi', 'sbref', 'sbref_brain', 'sbref_unwarped', 'sbref_fmap', 'mag2sbref_matrix',
                'fmap_unmasked', 'wm_seg']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['epi_brain', 'epi2sbref_matrix', 'stripped_epi', 'corrected_epi_mean']), name='outputnode')

    # Skull strip EPI  (try ComputeMask(BaseInterface))
    epi_bet = pe.Node(
        fsl.BET(mask=True, functional=True, frac=0.6), name="epi_bet")

    # do a standard flirt pre-alignment
    # flirt -ref ${vrefbrain} -in ${vepi} -dof ${dof} -omat ${vout}_init.mat
    flt_epi_sbref = pe.Node(fsl.FLIRT(dof=6, bins=640, cost_func='mutualinfo'),
                            name="EPI_2_SBRef_Brain_Affine_Transform")

    # Run MCFLIRT to get motion matrices
    # Motion Correction of the EPI with SBRef as Target
    motion_correct_epi = pe.Node(
        fsl.MCFLIRT(save_mats=True), name="Motion_Correction_EPI")
    # fslmaths ${vout}_fieldmaprads2str_dilated ${vout}_fieldmaprads2str
    # !!!! Don't need to do this since this just does the same thing as a "mv" command.
    # Just connect previous fugue node directly to subsequent flirt command.
    # run bbr to SBRef target with fieldmap and T1 seg in SBRef space
    # flirt -ref ${vrefhead} -in ${vepi} -dof ${dof} -cost bbr -wmseg
    # ${vout}_fast_wmseg -init ${vout}_init.mat -omat ${vout}.mat -out
    # ${vout}_1vol -schedule ${FSLDIR}/etc/flirtsch/bbr.sch -echospacing
    # ${dwell} -pedir ${pe_dir} -fieldmap ${vout}_fieldmaprads2str $wopt
    flt_bbr = pe.Node(fsl.FLIRT(dof=6, bins=640, cost_func='bbr', pedir=1, echospacing=dwell_time),
                      name="Flirt_BBR")
    flt_bbr.inputs.schedule = settings['fsl'].get(
        'flirt_bbr', op.join(os.getenv('FSLDIR'), 'etc/flirtsch/bbr.sch'))

    # make equivalent warp fields
    # convert_xfm -omat ${vout}_inv.mat -inverse ${vout}.mat
    invt_bbr = pe.Node(
        fsl.ConvertXFM(invert_xfm=True), name="BBR_Inverse_Transform")

    # convert_xfm -omat ${vout}_fieldmaprads2epi.mat -concat ${vout}_inv.mat
    # ${vout}_fieldmap2str.mat
    concat_mats = pe.Node(fsl.ConvertXFM(concat_xfm=True), name="BBR_Concat")

    # applywarp -i ${vout}_fieldmaprads_unmasked -r ${vepi}
    # --premat=${vout}_fieldmaprads2epi.mat -o ${vout}_fieldmaprads2epi
    aw_fmap_unmasked_epi = pe.Node(
        fsl.ApplyWarp(relwarp=True), name="Apply_Warp_Fmap_Unmasked_2_EPI")

    # fslmaths ${vout}_fieldmaprads2epi -abs -bin ${vout}_fieldmaprads2epi_mask
    fieldmaprads2epi_abs = pe.Node(
        fsl.UnaryMaths(operation='abs'), name="Abs_Fmap_2_EPI_Unmasked_Warp")
    fieldmaprads2epi_bin = pe.Node(
        fsl.UnaryMaths(operation='bin'), name="Binarize_Fmap_2_EPI_Unmasked_Warp")

    # fugue --loadfmap=${vout}_fieldmaprads2epi
    # --mask=${vout}_fieldmaprads2epi_mask
    # --saveshift=${vout}_fieldmaprads2epi_shift --unmaskshift
    # --dwell=${dwell} --unwarpdir=${fdir}
    fugue_shift = pe.Node(fsl.FUGUE(unwarp_direction='x', dwell_time=dwell_time,
                                    save_unmasked_shift=True), name="Fmap_Shift")

    # convertwarp -r ${vrefhead} -s ${vout}_fieldmaprads2epi_shift
    # --postmat=${vout}.mat -o ${vout}_warp --shiftdir=${fdir} --relout
    convert_fmap_shift = pe.MapNode(fsl.ConvertWarp(out_relwarp=True, shift_direction='x'),
                                    name="Convert_Fieldmap_Shift", iterfield=["premat"])

    split_epi = pe.Node(fsl.Split(dimension='t'), "Split_EPI")

    # applywarp -i ${vepi} -r ${vrefhead} -o ${vout} -w ${vout}_warp --interp=spline --rel
    aw_final = pe.MapNode(
        fsl.ApplyWarp(relwarp=True), name="Apply_Final_Warp", iterfield=["field_file", "in_file"])

    merge_epi = pe.Node(fsl.Merge(dimension='t'), "Merge_EPI")

    # BBR of Unwarped and SBRef-Reg
    epi_mean = pe.Node(fsl.MeanImage(dimension='T'), name="EPI_mean_volume")

    workflow.connect([
        (inputnode, epi_bet, [('epi', 'in_file')]),
        (inputnode, split_epi, [('epi', 'in_file')]),
        (inputnode, motion_correct_epi, [('sbref', 'ref_file')]),
        (inputnode, flt_epi_sbref, [('sbref_brain', 'reference')]),
        (inputnode, flt_bbr, [('sbref_unwarped', 'reference'),
                              ('sbref_fmap', 'fieldmap'),
                              ('wm_seg', 'wm_seg')]),
        (inputnode, concat_mats, [('mag2sbref_matrix', 'in_file')]),
        (inputnode, convert_fmap_shift, [('sbref_unwarped', 'reference')]),
        (inputnode, aw_final, [('sbref_unwarped', 'ref_file')]),
        (inputnode, aw_fmap_unmasked_epi, [('fmap_unmasked', 'in_file')]),
        (epi_bet, motion_correct_epi, [('out_file', 'in_file')]),
        (epi_bet, flt_epi_sbref, [('out_file', 'in_file')]),
        (epi_bet, flt_bbr, [('out_file', 'in_file')]),
        (epi_bet, aw_fmap_unmasked_epi, [('out_file', 'ref_file')]),
        (flt_epi_sbref, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
        (flt_bbr, invt_bbr, [('out_matrix_file', 'in_file')]),
        (invt_bbr, concat_mats, [('out_file', 'in_file2')]),
        (concat_mats, aw_fmap_unmasked_epi, [('out_file', 'premat')]),
        (aw_fmap_unmasked_epi, fieldmaprads2epi_abs, [('out_file', 'in_file')]),
        (fieldmaprads2epi_abs, fieldmaprads2epi_bin, [('out_file', 'in_file')]),
        (aw_fmap_unmasked_epi, fugue_shift, [('out_file', 'fmap_in_file')]),
        (fieldmaprads2epi_bin, fugue_shift, [('out_file', 'mask_file')]),
        (fugue_shift, convert_fmap_shift, [('shift_out_file', 'shift_in_file')]),
        (flt_bbr, convert_fmap_shift, [('out_matrix_file', 'postmat')]),
        (motion_correct_epi, convert_fmap_shift, [('mat_file', 'premat')]),
        (split_epi, aw_final, [('out_files', 'in_file')]),
        (convert_fmap_shift, aw_final, [('out_file', 'field_file')]),
        (aw_final, merge_epi, [('out_file', 'in_files')]),
        (merge_epi, epi_mean, [('merged_file', 'in_file')]),
        (epi_bet, outputnode, [('out_file', 'stripped_epi')]),
        (epi_mean, outputnode, [('out_file', 'corrected_epi_mean')])
    ])
    return workflow



