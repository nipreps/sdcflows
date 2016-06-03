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
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

from ..utils.misc import gen_list
from ..interfaces import ReadSidecarJSON
from .fieldmap import create_encoding_file
from ..viz import stripped_brain_overlay

# pylint: disable=R0914
def correction_workflow(name='EPIUnwarpWorkflow', settings=None):
    """ A workflow to correct EPI images """
    if settings is None:
        settings = {}

    dwell_time = settings['epi'].get('dwell_time', 0.000700012460221792)

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 'sbref', 'sbref_brain',
                              'sbref_unwarped', 'sbref_fmap',
                              'mag2sbref_matrix', 'fmap_unmasked', 'wm_seg',
                              't1_brain']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['epi_brain', 'epi2sbref_matrix',
                              'stripped_epi', 'corrected_epi_mean',
                              'merged_epi', 'stripped_epi_mask',
                              'epi_motion_params', 'bbr_sbref_2_t1']),
        name='outputnode'
    )

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
        (inputnode, flt_bbr, [('sbref_unwarped', 'in_file'),
                              ('sbref_fmap', 'fieldmap'),
                              ('wm_seg', 'wm_seg'),
                              ('t1_brain', 'reference')]),
        (inputnode, concat_mats, [('mag2sbref_matrix', 'in_file')]),
        (inputnode, convert_fmap_shift, [('sbref_unwarped', 'reference')]),
        (inputnode, aw_final, [('sbref_unwarped', 'ref_file')]),
        (inputnode, aw_fmap_unmasked_epi, [('fmap_unmasked', 'in_file')]),
        (inputnode, motion_correct_epi, [('epi', 'in_file')]),
        (epi_bet, flt_epi_sbref, [('out_file', 'in_file')]),
        (epi_bet, aw_fmap_unmasked_epi, [('out_file', 'ref_file')]),
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
        (merge_epi, outputnode, [('merged_file', 'merged_epi')]),
        (epi_bet, outputnode, [('out_file', 'stripped_epi')]),
        (epi_bet, outputnode, [('mask_file', 'stripped_epi_mask')]),
        (epi_mean, outputnode, [('out_file', 'corrected_epi_mean')]),
        (flt_bbr, outputnode, [('out_matrix_file', 'epi2sbref_matrix')]),
        (motion_correct_epi, outputnode, [('par_file', 'epi_motion_params')])
    ])
    return workflow

def _extract_wm(in_file):
    import os.path as op
    import nibabel as nb
    import numpy as np

    im = nb.load(in_file)
    data = im.get_data().astype(np.uint8)
    data[data != 3] = 0
    data[data > 0] = 1

    out_file = op.abspath('wm_mask.nii.gz')
    nb.Nifti1Image(data, im.get_affine(), im.get_header()).to_filename(out_file)
    return out_file
