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
from nipype.interfaces import ants
from nipype.interfaces import c3
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces import freesurfer as fs

from fmriprep.data import get_mni_template_ras
from fmriprep.interfaces import ReadSidecarJSON
from fmriprep.viz import stripped_brain_overlay
from fmriprep.workflows.fieldmap.se_pair_workflow import create_encoding_file
from fmriprep.workflows.sbref import _extract_wm


# pylint: disable=R0914
def epi_hmc(name='EPIHeadMotionCorrectionWorkflow', sbref_present=False, settings=None):
    """
    Performs :abbr:`HMC (head motion correction)` over the input
    :abbr:`EPI (echo-planar imaging)` image.
    """
    workflow = pe.Workflow(name=name)

    infields = ['epi', 't1_brain']
    if sbref_present:
        infields += ['sbref_brain']

    inputnode = pe.Node(niu.IdentityInterface(fields=infields),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['epi_brain', 'xforms']),
                         name='outputnode')

    # Reorient to RAS and skull-stripping
    split = pe.Node(fsl.Split(dimension='t'), name='SplitEPI')
    orient = pe.MapNode(fs.MRIConvert(out_type='niigz', out_orientation='RAS'),
                        iterfield=['in_file'], name='ReorientEPI')
    merge = pe.Node(fsl.Merge(dimension='t'), name='MergeEPI')
    bet = pe.Node(
        fsl.BET(mask=True, functional=True, frac=0.6),
        name="EPI_bet"
    )

    # Head motion correction (hmc)
    hmc = pe.Node(fsl.MCFLIRT(save_mats=True), name="EPI_hmc")

    workflow.connect([
        (inputnode, split, [('epi', 'in_file')]),
        (split, orient, [('out_files', 'in_file')]),
        (orient, merge, [('out_file', 'in_files')]),
        (merge, bet, [('merged_file', 'in_file')]),
        (bet, hmc, [('out_file', 'in_file')]),
        (hmc, outputnode, [('out_file', 'epi_brain'),
                           ('mat_file', 'xforms')]),
    ])

    # If we have an SBRef, it should be the reference,
    # align to mean volume otherwise
    if sbref_present:
        workflow.connect([
            (inputnode, hmc, [('sbref_brain', 'ref_file')]),
        ])
    else:
        hmc.inputs.mean_vol = True

    # Write corrected file in the designated output dir
    datasink = pe.Node(
        nio.DataSink(base_directory=op.join(settings['output_dir'], "images")),
        name="datasink",
        parameterization=False
    )

    workflow.connect([
        (hmc, datasink, [('out_file', '@epi_hmc')]),
    ])

    return workflow

def epi_mean_t1_registration(name='EPIMeanNormalization', settings=None):
    """
    Uses FSL FLIRT with the BBR cost function to find the transform that
    maps the EPI space into the T1-space
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 't1_brain', 't1_seg']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['mat_epi_to_t1']),
        name='outputnode'
    )

    epi_mean = pe.Node(fsl.MeanImage(dimension='T'), name="EPI_mean")

    # Extract wm mask from segmentation
    wm_mask = pe.Node(
        niu.Function(input_names=['in_file'], output_names=['out_file'],
        function=_extract_wm),
        name='WM_mask'
    )


    flt_bbr_init = pe.Node(fsl.FLIRT(dof=6, out_matrix_file='init.mat'), name="Flirt_BBR_init")

    flt_bbr = pe.Node(fsl.FLIRT(dof=6, cost_func='bbr'), name="Flirt_BBR")
    flt_bbr.inputs.schedule = settings['fsl'].get(
        'flirt_bbr', op.join(os.getenv('FSLDIR'), 'etc/flirtsch/bbr.sch'))

    # make equivalent warp fields
    invt_bbr = pe.Node(fsl.ConvertXFM(invert_xfm=True), name="Flirt_BBR_Inv")

    workflow.connect([
        (inputnode, epi_mean, [('epi', 'in_file')]),
        (inputnode, wm_mask, [('t1_seg', 'in_file')]),
        (inputnode, flt_bbr_init, [('t1_brain', 'reference')]),
        (epi_mean, flt_bbr_init, [('out_file', 'in_file')]),
        (flt_bbr_init, flt_bbr, [('out_matrix_file', 'in_matrix_file')]),
        (inputnode, flt_bbr, [('t1_brain', 'reference')]),
        (epi_mean, flt_bbr, [('out_file', 'in_file')]),
        (wm_mask, flt_bbr, [('out_file', 'wm_seg')]),
        (flt_bbr, invt_bbr, [('out_matrix_file', 'in_file')]),
        (flt_bbr, outputnode, [('out_matrix_file', 'mat_epi_to_t1')]),
        (invt_bbr, outputnode, [('out_file', 'mat_t1_to_epi')])
    ])

    # Plots for report
    png_sbref_t1 = pe.Node(niu.Function(
        input_names=["in_file", "overlay_file", "out_file"],
        output_names=["out_file"],
        function=stripped_brain_overlay),
        name="PNG_sbref_t1"
    )
    png_sbref_t1.inputs.out_file = "sbref_to_t1.png"

    datasink = pe.Node(nio.DataSink(base_directory=op.join(settings['work_dir'], "images")),
                       name="datasink", parameterization=False)

    workflow.connect([
        (flt_bbr, png_sbref_t1, [('out_file', 'overlay_file')]),
        (inputnode, png_sbref_t1, [('t1_seg', 'in_file')]),
        (png_sbref_t1, datasink, [('out_file', '@epi_to_t1')])
    ])

    return workflow

def epi_mni_transformation(name="EPIMNITransformation", settings=None):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'mat_epi_to_t1',
            't1_2_mni_forward_transform',
            'epi',
            't1',
            'hmc_xforms'
        ]),
        name='inputnode'
    )

    def _aslist(in_value):
        if isinstance(in_value, list):
            return in_value
        return [in_value]


    #  EPI to T1 transform matrix is from fsl, using c3 tools to convert to
    #  something ANTs will like.
    convert2itk = pe.Node(c3.C3dAffineTool(fsl2ras=True, itk_transform=True),
                       name='convert2itk')

    merge_transforms = pe.Node(niu.Merge(2), name="MergeTransforms")

    epi_to_T1_transform = pe.Node(ants.ApplyTransforms(), name="EPIToT1Transform")
    epi_t1_mni = pe.Node(ants.ApplyTransforms(), name="EPIToT1ToMNITransform")
    epi_t1_mni.inputs.reference_image = op.join(get_mni_template_ras(),
                                                'MNI152_T1_1mm.nii.gz')
    epi_to_mni_transform = pe.Node(ants.ApplyTransforms(), name="EPIToMNITransform")
    epi_to_mni_transform.inputs.reference_image = op.join(get_mni_template_ras(),
                                                          'MNI152_T1_1mm.nii.gz')
    epi_to_mni_transform.terminal_output = 'file'

    datasink = pe.Node(nio.DataSink(base_directory=op.join(settings['work_dir'], "images")),
                       name="datasink", parameterization=False)

    workflow.connect([
        (inputnode, convert2itk, [('mat_epi_to_t1', 'transform_file'),
                                  ('epi', 'source_file'),
                                  ('t1', 'reference_file')]),
        (convert2itk, merge_transforms, [(('itk_transform', _aslist), 'in1')]),
        (convert2itk, epi_to_T1_transform, [('itk_transform', 'transforms')]),
        (inputnode, epi_to_T1_transform, [('epi', 'input_image'),
                                          ('t1', 'reference_image')]),
        (inputnode, merge_transforms, [('t1_2_mni_forward_transform', 'in2')]),
        (merge_transforms, epi_to_mni_transform, [('out', 'transforms')]),
        (inputnode, epi_to_mni_transform, [('epi', 'input_image')]),
        (inputnode, epi_t1_mni, [('epi', 'input_image'),
                                 ('t1_2_mni_forward_transform', 'transforms')]),
        (epi_to_mni_transform, datasink, [('output_image', '@epi_mni')]),
        (epi_to_T1_transform, datasink, [('output_image', '@epi_t1')]),
        (epi_t1_mni, datasink, [('output_image', '@epi_t1_mni')])
    ])

    return workflow

# pylint: disable=R0914
def epi_unwarp(name='EPIUnwarpWorkflow', settings=None):
    """ A workflow to correct EPI images """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['epi', 'sbref_brain', 'fmap_fieldcoef', 'fmap_movpar',
                'fmap_mask', 'epi_brain']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['epi_correct', 'epi_mean']),
        name='outputnode'
    )

    # Read metadata
    meta = pe.MapNode(
        ReadSidecarJSON(fields=['TotalReadoutTime', 'PhaseEncodingDirection']),
        iterfield=['in_file'],
        name='metadata'
    )

    encfile = pe.Node(interface=niu.Function(
        input_names=['fieldmaps', 'in_dict'], output_names=['parameters_file'],
        function=create_encoding_file), name='TopUp_encfile', updatehash=True)

    fslsplit = pe.Node(fsl.Split(dimension='t'), name='EPI_split')

    # Now, we cannot use the LSR method
    unwarp_epi = pe.MapNode(fsl.ApplyTOPUP(method='jac', in_index=[1]),
                            iterfield=['in_files'], name='TopUpApply')

    # Merge back
    fslmerge = pe.Node(fsl.Merge(dimension='t'), name='EPI_corr_merge')

    # Compute mean
    epi_mean = pe.Node(fsl.MeanImage(dimension='T'), name="EPI_mean")

    workflow.connect([
        (inputnode, meta, [('epi', 'in_file')]),
        (inputnode, encfile, [('epi', 'fieldmaps')]),
        (inputnode, fslsplit, [('epi_brain', 'in_file')]),
        (meta, encfile, [('out_dict', 'in_dict')]),
        (inputnode, unwarp_epi, [('fmap_fieldcoef', 'in_topup_fieldcoef'),
                                 ('fmap_movpar', 'in_topup_movpar')]),
        (encfile, unwarp_epi, [('parameters_file', 'encoding_file')]),
        (fslsplit, unwarp_epi, [('out_files', 'in_files')]),
        (unwarp_epi, fslmerge, [('out_corrected', 'in_files')]),
        (fslmerge, epi_mean, [('merged_file', 'in_file')]),
        (fslmerge, outputnode, [('merged_file', 'epi_correct')]),
        (epi_mean, outputnode, [('out_file', 'epi_mean')])
    ])

    # Plot result
    png_epi_corr= pe.Node(niu.Function(
        input_names=["in_file", "overlay_file", "out_file"], output_names=["out_file"],
        function=stripped_brain_overlay), name="PNG_epi_corr")
    png_epi_corr.inputs.out_file = "corrected_EPI.png"

    datasink = pe.Node(nio.DataSink(base_directory=op.join(settings['output_dir'], "images")),
                       name="datasink", parameterization=False)

    workflow.connect([
        (epi_mean, png_epi_corr, [('out_file', 'overlay_file')]),
        (inputnode, png_epi_corr, [('fmap_mask', 'in_file')]),
        (png_epi_corr, datasink, [('out_file', '@corrected_EPI')])
    ])

    return workflow
