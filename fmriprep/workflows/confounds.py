#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import utility as niu, fsl
from niworkflows.nipype.interfaces.nilearn import SignalExtraction
from niworkflows.nipype.algorithms import confounds as nac

from niworkflows.interfaces import segmentation as nws
from niworkflows.interfaces.masks import ACompCorRPT, TCompCorRPT

from ..interfaces import (
    TPM2ROI, ConcatROIs, CombineROIs, AddTSVHeader, GatherConfounds, ICAConfounds
)


def init_bold_confs_wf(bold_file_size_gb, use_aroma, ignore_aroma_err, metadata,
                       name="bold_confs_wf"):
    ''' All input fields are required.

    Calculates global regressor and tCompCor
        from motion-corrected fMRI ('inputnode.fmri_file').
    Calculates DVARS from the fMRI and an EPI brain mask ('inputnode.bold_mask')
    Calculates frame displacement from MCFLIRT movement parameters ('inputnode.movpar_file')
    Calculates segment regressors and aCompCor
        from the fMRI and a white matter/gray matter/CSF segmentation ('inputnode.t1_seg'), after
        applying the transform to the images. Transforms should be fsl-formatted.
    Calculates noise components identified from ICA-AROMA (if ``use_aroma=True``)
    Saves the confounds in a file ('outputnode.confounds_file')'''

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['fmri_file', 'movpar_file', 't1_tpms', 'bold_mask', 'bold_mni', 'bold_mask_mni']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['confounds_file', 'confounds_list', 'acompcor_report', 'tcompcor_report',
                'ica_aroma_report', 'aroma_noise_ics', 'melodic_mix',
                'nonaggr_denoised_file']),
        name='outputnode')

    # ICA-AROMA
    if use_aroma:
        ica_aroma_wf = init_ica_aroma_wf(name='ica_aroma_wf',
                                         ignore_aroma_err=ignore_aroma_err)

    # DVARS
    dvars = pe.Node(nac.ComputeDVARS(save_all=True, remove_zerovariance=True),
                    name="dvars", mem_gb=bold_file_size_gb * 3)

    # Frame displacement
    fdisp = pe.Node(nac.FramewiseDisplacement(parameter_source="SPM"),
                    name="fdisp", mem_gb=bold_file_size_gb * 3)

    # CompCor
    non_steady_state = pe.Node(nac.NonSteadyStateDetector(), name='non_steady_state')
    tcompcor = pe.Node(TCompCorRPT(components_file='tcompcor.tsv',
                                   generate_report=True,
                                   pre_filter='cosine',
                                   save_pre_filter=True,
                                   percentile_threshold=.05),
                       name="tcompcor", mem_gb=bold_file_size_gb * 3)
    acompcor = pe.Node(ACompCorRPT(components_file='acompcor.tsv',
                                   pre_filter='cosine',
                                   save_pre_filter=True,
                                   generate_report=True),
                       name="acompcor", mem_gb=bold_file_size_gb * 3)

    csf_roi = pe.Node(TPM2ROI(erode_mm=0, mask_erode_mm=30), name='csf_roi')
    wm_roi = pe.Node(TPM2ROI(erode_mm=6, mask_erode_mm=10), name='wm_roi')
    merge_rois = pe.Node(niu.Merge(2), name='merge_rois', run_without_submit=True, mem_gb=0.01)
    combine_rois = pe.Node(CombineROIs(), name='combine_rois')
    concat_rois = pe.Node(ConcatROIs(), name='concat_rois')

    # Global and segment regressors
    signals = pe.Node(SignalExtraction(detrend=True,
                                       class_labels=["WhiteMatter", "GlobalSignal"]),
                      name="signals", mem_gb=bold_file_size_gb * 3)

    # Arrange confounds
    add_header = pe.Node(AddTSVHeader(columns=["X", "Y", "Z", "RotX", "RotY", "RotZ"]),
                         name="add_header", mem_gb=0.01, run_without_submit=True)
    concat = pe.Node(GatherConfounds(), name="concat", mem_gb=0.01, run_without_submit=True)

    # Set TR if present
    if 'RepetitionTime' in metadata:
        tcompcor.inputs.repetition_time = metadata['RepetitionTime']
        acompcor.inputs.repetition_time = metadata['RepetitionTime']

    def _pick_csf(files):
        return files[0]

    def _pick_wm(files):
        return files[2]

    workflow = pe.Workflow(name=name)
    workflow.connect([
        # connect inputnode to each non-anatomical confound node
        (inputnode, dvars, [('fmri_file', 'in_file'),
                            ('bold_mask', 'in_mask')]),
        (inputnode, fdisp, [('movpar_file', 'in_file')]),
        (inputnode, non_steady_state, [('fmri_file', 'in_file')]),
        (inputnode, tcompcor, [('fmri_file', 'realigned_file')]),

        (non_steady_state, tcompcor, [('n_volumes_to_discard', 'ignore_initial_volumes')]),
        (non_steady_state, acompcor, [('n_volumes_to_discard', 'ignore_initial_volumes')]),

        (inputnode, csf_roi, [(('t1_tpms', _pick_csf), 'in_file')]),
        (inputnode, csf_roi, [('bold_mask', 'in_mask')]),
        (csf_roi, tcompcor, [('eroded_mask', 'mask_files')]),

        (inputnode, wm_roi, [(('t1_tpms', _pick_wm), 'in_file')]),
        (inputnode, wm_roi, [('bold_mask', 'in_mask')]),

        (csf_roi, merge_rois, [('roi_file', 'in1')]),
        (wm_roi, merge_rois, [('roi_file', 'in2')]),
        (merge_rois, combine_rois, [('out', 'in_files')]),
        (inputnode, combine_rois, [('fmri_file', 'ref_header')]),

        # anatomical confound: aCompCor.
        (inputnode, acompcor, [('fmri_file', 'realigned_file')]),
        (combine_rois, acompcor, [('out_file', 'mask_files')]),

        (wm_roi, concat_rois, [('roi_file', 'in_file')]),
        (inputnode, concat_rois, [('bold_mask', 'in_mask')]),
        (inputnode, concat_rois, [('fmri_file', 'ref_header')]),

        # anatomical confound: signal extraction
        (concat_rois, signals, [('out_file', 'label_files')]),
        (inputnode, signals, [('fmri_file', 'in_file')]),

        # connect the confound nodes to the concatenate node
        (inputnode, add_header, [('movpar_file', 'in_file')]),
        (signals, concat, [('out_file', 'signals')]),
        (dvars, concat, [('out_all', 'dvars')]),
        (fdisp, concat, [('out_file', 'fd')]),
        (tcompcor, concat, [('components_file', 'tcompcor'),
                            ('pre_filter_file', 'cos_basis')]),
        (acompcor, concat, [('components_file', 'acompcor')]),
        (add_header, concat, [('out_file', 'motion')]),

        (concat, outputnode, [('confounds_file', 'confounds_file'),
                              ('confounds_list', 'confounds_list')]),
        (acompcor, outputnode, [('out_report', 'acompcor_report')]),
        (tcompcor, outputnode, [('out_report', 'tcompcor_report')]),
    ])

    if use_aroma:
        workflow.connect([
            (inputnode, ica_aroma_wf, [('bold_mni', 'inputnode.bold_mni'),
                                       ('bold_mask_mni', 'inputnode.bold_mask_mni'),
                                       ('movpar_file', 'inputnode.movpar_file')]),
            (ica_aroma_wf, concat,
                [('outputnode.aroma_confounds', 'aroma')]),
            (ica_aroma_wf, outputnode,
                [('outputnode.out_report', 'ica_aroma_report'),
                 ('outputnode.aroma_noise_ics', 'aroma_noise_ics'),
                 ('outputnode.melodic_mix', 'melodic_mix'),
                 ('outputnode.nonaggr_denoised_file', 'nonaggr_denoised_file')])
        ])
    return workflow


def init_ica_aroma_wf(name='ica_aroma_wf', ignore_aroma_err=False):
    '''
    From: https://github.com/rhr-pruim/ICA-AROMA
    Description:
    ICA-AROMA (i.e. ‘ICA-based Automatic Removal Of Motion Artifacts’) concerns
    a data-driven method to identify and remove motion-related independent
    components from fMRI data.

    Preconditions/Assumptions:
    The input fmri bold file is in standard space
    (for ease of interfacing with the original ICA-AROMA code)

    Steps:
    1) smooth data using SUSAN
    2) run melodic outside of ICA-AROMA to generate the report
    3) run ICA-AROMA
    4) print identified motion components (aggressive) to tsv
    5) pass classified_motion_ICs and melodic_mix for user to complete nonaggr denoising
    '''
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_mni', 'movpar_file', 'bold_mask_mni']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['aroma_confounds', 'out_report',
                'aroma_noise_ics', 'melodic_mix',
                'nonaggr_denoised_file']), name='outputnode')

    calc_median_val = pe.Node(fsl.ImageStats(op_string='-k %s -p 50'), name='calc_median_val')
    calc_bold_mean = pe.Node(fsl.MeanImage(), name='calc_bold_mean')

    def getusans_func(image, thresh):
        return [tuple([image, thresh])]
    getusans = pe.Node(niu.Function(
        function=getusans_func, input_names=['image', 'thresh'], output_names=['usans']),
        name='getusans', run_without_submit=True, mem_gb=0.01)

    smooth = pe.Node(fsl.SUSAN(fwhm=6.0), name='smooth')

    # melodic node
    melodic = pe.Node(
        fsl.MELODIC(no_bet=True,
                    no_mm=True),
        name="melodic")

    # ica_aroma node
    ica_aroma = pe.Node(nws.ICA_AROMARPT(denoise_type='nonaggr',
                                         generate_report=True), name='ica_aroma')

    # extract the confound ICs from the results
    ica_aroma_confound_extraction = pe.Node(ICAConfounds(ignore_aroma_err=ignore_aroma_err),
                                            name='ica_aroma_confound_extraction')

    def _getbtthresh(medianval):
        return 0.75 * medianval

    # connect the nodes
    workflow.connect([
        # Connect input nodes to complete smoothing
        (inputnode, calc_median_val, [('bold_mni', 'in_file'),
                                      ('bold_mask_mni', 'mask_file')]),
        (inputnode, calc_bold_mean, [('bold_mni', 'in_file')]),
        (calc_bold_mean, getusans, [('out_file', 'image')]),
        (calc_median_val, getusans, [('out_stat', 'thresh')]),
        (inputnode, smooth, [('bold_mni', 'in_file')]),
        (getusans, smooth, [('usans', 'usans')]),
        (calc_median_val, smooth, [(('out_stat', _getbtthresh), 'brightness_threshold')]),
        # connect smooth to melodic
        (smooth, melodic, [('smoothed_file', 'in_files')]),
        (inputnode, melodic, [('bold_mask_mni', 'mask')]),
        # connect nodes to ICA-AROMA
        (smooth, ica_aroma, [('smoothed_file', 'in_file')]),
        (inputnode, ica_aroma, [('bold_mask_mni', 'report_mask'),
                                ('movpar_file', 'motion_parameters')]),
        (melodic, ica_aroma, [('out_dir', 'melodic_dir')]),
        # generate tsvs from ICA-AROMA
        (ica_aroma, ica_aroma_confound_extraction, [('out_dir', 'in_directory')]),
        # output for processing and reporting
        (ica_aroma_confound_extraction, outputnode, [('aroma_confounds', 'aroma_confounds'),
                                                     ('aroma_noise_ics', 'aroma_noise_ics'),
                                                     ('melodic_mix', 'melodic_mix')]),
        # TODO change melodic report to reflect noise and non-noise components
        (ica_aroma, outputnode, [('out_report', 'out_report'),
                                 ('nonaggr_denoised_file', 'nonaggr_denoised_file')]),
    ])

    return workflow
