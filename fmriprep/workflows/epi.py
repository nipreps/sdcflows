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
import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import ants
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu
from nipype.interfaces import fsl
from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection

from fmriprep.data import get_mni_template
from fmriprep.interfaces import ReadSidecarJSON
from fmriprep.utils.misc import gen_list
from fmriprep.viz import stripped_brain_overlay
from fmriprep.workflows.fieldmap.se_pair_workflow import create_encoding_file
from fmriprep.workflows.sbref import _extract_wm


# pylint: disable=R0914
def epi_hmc(subject_data, name='EPIHeadMotionCorrectionWorkflow', settings=None):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 'sbref_brain', 't1_brain']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['epi_brain']),
        name='outputnode'
    )

    epi_bet = pe.Node(
        fsl.BET(mask=True, functional=True, frac=0.6),
        name="EPI_bet"
    )
    epi_hmc = pe.Node(fsl.MCFLIRT(save_mats=True), name="EPI_hmc")

    workflow.connect([
        (inputnode, epi_bet, [('epi', 'in_file')]),
        (epi_bet, epi_hmc, [('out_file', 'in_file')]),
        (epi_hmc, outputnode, [('out_file', 'epi_brain')]),
    ])

    if subject_data['sbref'] == []: 
        epi_mean = pe.Node(fsl.MeanImage(dimension='T'), name="EPI_mean")
        workflow.connect([
            (inputnode, epi_mean, [('epi', 'in_file')]),
            (epi_mean, epi_hmc, [('out_file', 'ref_file')]),
        ])
    else:
        workflow.connect([
            (inputnode, epi_hmc, [('sbref_brain', 'ref_file')]),
        ])
        
    datasink = pe.Node(
        nio.DataSink(base_directory=op.join(settings['output_dir'], "images")),
        name="datasink",
        parameterization=False
    )

    workflow.connect([
        (epi_hmc, datasink, [('out_file', '@epi_hmc')]),
    ])

    return workflow

def epi_mean_t1_registration(name='EPIMeanNormalization', settings=None):
    """
    Uses FSL FLIRT with the BBR cost function to find the transform that
    maps the EPI space into the T1-space
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=['epi', 't1', 't1_seg']),
        name='inputnode'
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=['mat_epi_to_t1']),
        name='outputnode'
    )

    # 5. T1w to MNI registration
    epi_to_t1 = pe.Node(ants.Registration(float=True), name="EPI_To_T1_Registration")
    epi_to_t1.inputs.fixed_image = op.join(get_mni_template(), 'MNI152_T1_1mm.nii.gz')
    epi_to_t1.inputs.fixed_image_mask = op.join(
        get_mni_template(), 'MNI152_T1_1mm_brain_mask.nii.gz')

    # Hack to avoid re-running ANTs all the times
    grabber_interface = nio.JSONFileGrabber()
    setattr(grabber_interface, '_always_run', False)
    epi_to_t1_params = pe.Node(grabber_interface, name='t1_2_mni_params')
    epi_to_t1_params.inputs.in_file = (
        pkgr.resource_filename('fmriprep', 'data/registration_settings.json')
    )


    datasink = pe.Node(nio.DataSink(base_directory=op.join(settings['work_dir'], "images")),
                       name="datasink", parameterization=False)

    workflow.connect([
        (inputnode, epi_to_t1, [('t1', 'fixed_image'),
                                ('epi', 'moving_image')]),
        (epi_to_t1, outputnode, [('forward_transforms', 'mat_epi_to_t1')]),
        (epi_to_t1, datasink, [('warped_image', '@warped_epi_to_t1')])
    ])

    return workflow

def epi_mni_transformation(name="EPIMNITransformation", settings=None):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=[
            'mat_epi_to_t1',
            't1_2_mni_forward_transform',
            'epi',
            't1'
        ]),
        name='inputnode'
    )

    merge_transforms = pe.Node(niu.Merge(2), name="MergeTransforms")

    epi_to_mni_transform = pe.Node(ants.ApplyTransforms(), name="EPIToMNITransform")
    epi_to_mni_transform.reference_image = op.join(get_mni_template(), 
                                                   'MNI152_T1_1mm.nii.gz')
    epi_to_mni_transform.terminal_output = 'file'

    datasink = pe.Node(nio.DataSink(base_directory=op.join(settings['work_dir'], "images")),
                       name="datasink", parameterization=False)

    workflow.connect([
        (inputnode, merge_transforms, [('mat_epi_to_t1', 'in1')]),
        (inputnode, merge_transforms, [('t1_2_mni_forward_transform', 'in2')]),
        (merge_transforms, epi_to_mni_transform, [('out', 'transforms')]),
        (inputnode, epi_to_mni_transform, [('epi', 'input_image')]),
        (epi_to_mni_transform, datasink, [('output_image', '@epi_mni')]),
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
