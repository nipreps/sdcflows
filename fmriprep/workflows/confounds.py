#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''
from __future__ import print_function, division, absolute_import, unicode_literals

from nipype.interfaces import utility
from nipype.algorithms import confounds
from nipype.pipeline import engine as pe
from niworkflows.interfaces.masks import ACompCorRPT, TCompCorRPT

from nipype.interfaces.nilearn import SignalExtraction
from fmriprep.interfaces.utils import prepare_roi_from_probtissue


def init_discover_wf(bold_file_size_gb, name="discover_wf"):
    ''' All input fields are required.

    Calculates global regressor and tCompCor
        from motion-corrected fMRI ('inputnode.fmri_file').
    Calculates DVARS from the fMRI and an EPI brain mask ('inputnode.epi_mask')
    Calculates frame displacement from MCFLIRT movement parameters ('inputnode.movpar_file')
    Calculates segment regressors and aCompCor
        from the fMRI and a white matter/gray matter/CSF segmentation ('inputnode.t1_seg'), after
        applying the transform to the images. Transforms should be fsl-formatted.

    Saves the confounds in a file ('outputnode.confounds_file')'''

    inputnode = pe.Node(utility.IdentityInterface(
        fields=['fmri_file', 'movpar_file', 't1_tpms', 'epi_mask', 'epi_ref', 't1_head', 't1_brain']),
        name='inputnode')
    outputnode = pe.Node(utility.IdentityInterface(
        fields=['confounds_file', 'acompcor_report', 'tcompcor_report']),
        name='outputnode')

    #AROMA
    if use_AROMA:
        ica_aroma_wf = init_ica_aroma_wf(denoise_strategy=denoise_strategy)

    # DVARS
    dvars = pe.Node(confounds.ComputeDVARS(save_all=True, remove_zerovariance=True),
                    name="dvars")
    dvars.interface.estimated_memory_gb = bold_file_size_gb * 3
    # Frame displacement
    frame_displace = pe.Node(confounds.FramewiseDisplacement(parameter_source="SPM"),
                             name="frame_displace")
    frame_displace.interface.estimated_memory_gb = bold_file_size_gb * 3
    # CompCor
    tcompcor = pe.Node(TCompCorRPT(components_file='tcompcor.tsv',
                                   generate_report=True,
                                   percentile_threshold=.05),
                       name="tcompcor")
    tcompcor.interface.estimated_memory_gb = bold_file_size_gb * 3

    CSF_roi = pe.Node(utility.Function(function=prepare_roi_from_probtissue,
                                       output_names=['roi_file', 'eroded_mask']),
                      name='CSF_roi')
    CSF_roi.inputs.erosion_mm = 0
    CSF_roi.inputs.epi_mask_erosion_mm = 30

    WM_roi = pe.Node(utility.Function(function=prepare_roi_from_probtissue,
                                      output_names=['roi_file', 'eroded_mask']),
                     name='WM_roi')
    WM_roi.inputs.erosion_mm = 6
    WM_roi.inputs.epi_mask_erosion_mm = 10

    def concat_rois_func(in_WM, in_mask, ref_header):
        import os
        import nibabel as nb
        from nilearn.image import resample_to_img

        WM_nii = nb.load(in_WM)
        mask_nii = nb.load(in_mask)

        # we have to do this explicitly because of potential differences in
        # qform_code between the two files that prevent SignalExtraction to do
        # the concatenation
        concat_nii = nb.funcs.concat_images([resample_to_img(WM_nii,
                                                             mask_nii,
                                                             interpolation='nearest'),
                                             mask_nii])
        concat_nii = nb.Nifti1Image(concat_nii.get_data(),
                                    nb.load(ref_header).affine,
                                    nb.load(ref_header).header)
        concat_nii.to_filename("concat.nii.gz")
        return os.path.abspath("concat.nii.gz")

    concat_rois = pe.Node(utility.Function(function=concat_rois_func), name='concat_rois')

    # Global and segment regressors
    signals = pe.Node(SignalExtraction(detrend=True,
                                       class_labels=["WhiteMatter", "GlobalSignal"]),
                      name="signals")
    signals.interface.estimated_memory_gb = bold_file_size_gb * 3

    def combine_rois(in_CSF, in_WM, ref_header):
        import os
        import numpy as np
        import nibabel as nb

        CSF_nii = nb.load(in_CSF)
        CSF_data = CSF_nii.get_data()

        WM_nii = nb.load(in_WM)
        WM_data = WM_nii.get_data()

        combined = np.zeros_like(WM_data)

        combined[WM_data != 0] = 1
        combined[CSF_data != 0] = 1

        # we have to do this explicitly because of potential differences in
        # qform_code between the two files that prevent aCompCor to work
        new_nii = nb.Nifti1Image(combined, nb.load(ref_header).affine,
                                 nb.load(ref_header).header)
        new_nii.to_filename("logical_or.nii.gz")
        return os.path.abspath("logical_or.nii.gz")

    combine_rois = pe.Node(utility.Function(function=combine_rois), name='combine_rois')

    acompcor = pe.Node(ACompCorRPT(components_file='acompcor.tsv',
                                   generate_report=True),
                       name="acompcor")
    acompcor.interface.estimated_memory_gb = bold_file_size_gb * 3

    # misc utilities
    concat = pe.Node(utility.Function(function=_gather_confounds), name="concat")

    def pick_csf(files):
        return files[0]

    def pick_wm(files):
        return files[2]

    def add_header_func(in_file):
        import numpy as np
        import pandas as pd
        import os
        from sys import version_info
        PY3 = version_info[0] > 2

        data = np.loadtxt(in_file)

        df = pd.DataFrame(data, columns=["X", "Y", "Z", "RotX", "RotY", "RotZ"])
        df.to_csv("motion.tsv", sep="\t" if PY3 else '\t'.encode(), index=None)

        return os.path.abspath("motion.tsv")

    add_header = pe.Node(utility.Function(function=add_header_func), name="add_header")

    workflow = pe.Workflow(name=name)
    workflow.connect([
        # connect inputnode to each non-anatomical confound node
        (inputnode, dvars, [('fmri_file', 'in_file'),
                            ('epi_mask', 'in_mask')]),
        (inputnode, frame_displace, [('movpar_file', 'in_file')]),
        (inputnode, tcompcor, [('fmri_file', 'realigned_file')]),

        (inputnode, CSF_roi, [(('t1_tpms', pick_csf), 'in_file')]),
        (inputnode, CSF_roi, [('epi_mask',  'epi_mask')]),
        (CSF_roi, tcompcor, [('eroded_mask', 'mask_files')]),

        (inputnode, WM_roi, [(('t1_tpms', pick_wm), 'in_file')]),
        (inputnode, WM_roi, [('epi_mask', 'epi_mask')]),

        (CSF_roi, combine_rois, [('roi_file', 'in_CSF')]),
        (WM_roi, combine_rois, [('roi_file', 'in_WM')]),
        (inputnode, combine_rois, [('fmri_file', 'ref_header')]),

        # anatomical confound: aCompCor.
        (inputnode, acompcor, [('fmri_file', 'realigned_file')]),
        (combine_rois, acompcor, [('out', 'mask_files')]),

        (WM_roi, concat_rois, [('roi_file', 'in_WM')]),
        (inputnode, concat_rois, [('epi_mask', 'in_mask')]),
        (inputnode, concat_rois, [('fmri_file', 'ref_header')]),

        # anatomical confound: signal extraction
        (concat_rois, signals, [('out', 'label_files')]),
        (inputnode, signals, [('fmri_file', 'in_file')]),

        # connect the confound nodes to the concatenate node
        (signals, concat, [('out_file', 'signals')]),
        (dvars, concat, [('out_all', 'dvars')]),
        (frame_displace, concat, [('out_file', 'frame_displace')]),
        (tcompcor, concat, [('components_file', 'tcompcor')]),
        (acompcor, concat, [('components_file', 'acompcor')]),
        (inputnode, add_header, [('movpar_file', 'in_file')]),
        (add_header, concat, [('out', 'motion')]),

        (concat, outputnode, [('out', 'confounds_file')]),
        (acompcor, outputnode, [('out_report', 'acompcor_report')]),
        (tcompcor, outputnode, [('out_report', 'tcompcor_report')]),
    ])
    if use_AROMA:
        workflow.connect([
            (inputnode,ica_aroma_wf,
                [('t1_brain', 't1_brain'),
                 ('t1_head', 't1_head'),
                 ('epi_ref', 'epi_ref'),
                 ('fmri_file', 'fmri_file'),
                 ('movpar_file', 'movpar_file'),
                 ('epi_mask', 'epi_mask')]),
            (ica_aroma_wf,concat,
                [('motion_ICs','aroma')])

            ])
    return workflow


def _gather_confounds(signals=None, dvars=None, frame_displace=None,
                      tcompcor=None, acompcor=None, motion=None, aroma=None):
    ''' load confounds from the filenames, concatenate together horizontally, and re-save '''
    import pandas as pd
    import os.path as op

    def less_breakable(a_string):
        ''' hardens the string to different envs (i.e. case insensitive, no whitespace, '#' '''
        return ''.join(a_string.split()).strip('#')

    def _adjust_indices(left_df, right_df):
        # This forces missing values to appear at the beggining of the DataFrame
        # instead of the end
        index_diff = len(left_df.index) - len(right_df.index)
        if index_diff > 0:
            right_df.index = range(index_diff,
                                   len(right_df.index) + index_diff)
        elif index_diff < 0:
            left_df.index = range(-index_diff,
                                  len(left_df.index) - index_diff)

    all_files = [confound for confound in [signals, dvars, frame_displace,
                                           tcompcor, acompcor, motion]
                 if confound is not None]

    confounds_data = pd.DataFrame()
    for file_name in all_files:  # assumes they all have headings already
        new = pd.read_csv(file_name, sep="\t")
        for column_name in new.columns:
            new.rename(columns={column_name: less_breakable(column_name)},
                       inplace=True)

        _adjust_indices(confounds_data, new)
        confounds_data = pd.concat((confounds_data, new), axis=1)

    combined_out = op.abspath('confounds.tsv')
    confounds_data.to_csv(combined_out, sep=str("\t"), index=False,
                          na_rep="n/a")

    return combined_out


def reverse_order(inlist):
    ''' if a list, return the list in reversed order; else it is a single item, return it.'''
    if isinstance(inlist, list):
        inlist.reverse()
    return inlist

def init_ica_aroma_wf(name='ica_aroma_wf',denoise_strategy='nonaggr'):
    #standard image for ICA-AROMA
    mni_image = fsl.Info.standard_image('MNI152_T1_2mm.nii.gz')

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(util.IdentityInterface(
        fields=['t1_brain','t1_head','epi_ref',
                'fmri_file','movpar_file','epi_mask']),
                        name='inputnode')

    outputnode = pe.Node(util.IdentityInterface(
        fields=['motion_ICs']), name='outputnode')

    #flirt node t1 to mni
    aroma_t1tomni_flirt = pe.Node(fsl.preprocess.FLIRT(
            reference=mni_image),name='aroma_t1tomni_flirt')
    
    #fnirt node t1 to mni
    aroma_t1tomni_fnirt = pe.Node(fsl.preprocess.FNIRT(
            ref_file=mni_image,
            field_file=True),
            name='aroma_t1tomni_fnirt')

    #flirt node epi to t1
    aroma_epitot1_flirt = pe.Node(fsl.preprocess.FLIRT(),
            name='aroma_epitot1_flirt')

    #smoothing node

    #functions to help set SUSAN 
    def getbtthresh(medianval):
        return 0.75 * medianval

    def getusans_func(image,thresh):
        return [tuple([image,thresh])]
    median_val = pe.Node(fsl.ImageStats(op_string='-k %s -p 50'),
                              name='median_val')
    
    getusans = pe.Node(Function(input_names=['image','thresh'],
                                     output_names=['usans'],
                                     function=getusans_func),
                            name='getusans')

    brightness_threshold = pe.Node(Function(input_names=['medianval'],
                                     output_names=['thresh'],
                                     function=getbtthresh),
                                        name='brightness_threshold')

    smooth = pe.Node(fsl.SUSAN(fwhm=2.5),
                          name='smooth')
    
    #ica_aroma node
    ica_aroma = pe.Node(fsl.ICA_AROMA.ICA_AROMA(denoise_type=denoise_strategy),
                             name='ica_aroma')
    
    #set the output directory manually (until pull request #2056 is merged.)
    ica_out_dir = ica_aroma.output_dir()
    ica_aroma.set_input('out_dir',ica_out_dir)

    #extract the confound ICs from the results
    def get_ica_confounds(ica_out_dir):
        import os
        import numpy as np

        melodic_mix = os.path.join(ica_out_dir,'melodic.ica/melodic_mix')
        motion_ICs = os.path.join(ica_out_dir,'classified_motion_ICs.txt')

        #-1 since lists start at index 0
        motion_ic_indices = np.loadtxt(motion_ICs,dtype=int,delimiter=',')-1
        melodic_mix_arr = np.loadtxt(melodic_mix,ndmin=2)

        #transpose melodic_mix_arr so indices refer to the correct dimension
        motion_ic_arr = np.asarray([melodic_mix_arr.T[x] for x in motion_ic_indices])

        confound_txt = 'confound_ics.txt'

        #transpose result back so the number of lines in the file 
        #is equal to the length of the fmri run.
        np.savetxt(confound_txt,motion_ic_arr.T,fmt='%.10f')

        return confound_txt

    ica_confound = pe.Node(Function(input_names=['ica_out_dir'],
                                    output_names=['confound_txt'],
                                    function=get_ica_confounds),
                            name='ic_confound')

    #connect the nodes
    workflow.connect([
        #Connect nodes to complete smoothing
        (inputnode,median_val,
            [('epi_mask','mask_file'),
            ('fmri_file','in_file')]),
        (median_val, brightness_threshold,
            [('out_stat','medianval')]),       
        (inputnode, getusans,
            [('epi_ref','image')]),
        (median_val, getusans,
            [('out_stat','thresh')]),
        (inputnode,smooth,
            [('fmri_file','in_file')]),
        (getusans,smooth,
            [('usans', 'usans')]),
        (brightness_threshold, smooth,
            [('thresh','brightness_threshold')]),
        #connect nodes to ICA-AROMA
        (smooth, ica_aroma,
            [('smoothed_file','in_file')]),
        (inputnode, ica_aroma,
            [('movpar_file','motion_parameters'),
             ('epi_mask','mask')]),
        #generate transforms for ICA-AROMA
        (inputnode, aroma_t1tomni_flirt, 
            [('t1_brain', 'in_file')]),
        (inputnode, aroma_t1tomni_fnirt,
            [('t1_head', 'in_file')]),
        (aroma_t1tomni_flirt, aroma_t1tomni_fnirt,
            [('out_matrix_file','affine_file')]),
        (inputnode, aroma_epitot1_flirt,
            [('epi_ref','in_file'),
             ('t1_brain','reference')]),
        #Connect transforms to ICA-AROMA
        (aroma_t1tomni_fnirt, ica_aroma,
            [('field_file','fnirt_warp_file')]),
        (aroma_epitot1_flirt, ica_aroma,
            [('out_matrix_file','mat_file')]),
        #output for processing
        (ica_aroma,ica_confound,
            [('out_dir','ica_out_dir')]),
        (ica_confound,outputnode,
            [('confound_txt','motion_ICs')])
        ])

    return workflow