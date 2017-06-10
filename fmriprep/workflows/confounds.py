#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''
from __future__ import print_function, division, absolute_import, unicode_literals
from niworkflows.nipype.interfaces import utility
from niworkflows.nipype.interfaces import fsl
from niworkflows.nipype.algorithms import confounds
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces.fsl import ICA_AROMA as aroma
from niworkflows.interfaces.masks import ACompCorRPT, TCompCorRPT
from niworkflows.interfaces import segmentation as nws


from niworkflows.nipype.interfaces.nilearn import SignalExtraction
from fmriprep.interfaces.utils import prepare_roi_from_probtissue


def init_discover_wf(bold_file_size_gb, name="discover_wf",use_aroma=False):
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
        fields=['fmri_file', 'movpar_file', 't1_tpms', 'epi_mask', 'epi_mni', 'epi_mask_mni']),
        name='inputnode')
    outputnode = pe.Node(utility.IdentityInterface(
        fields=['confounds_file', 'acompcor_report', 'tcompcor_report', 'ica_aroma_report']),
        name='outputnode')

    #AROMA

    if use_aroma:
        ica_aroma_wf = init_ica_aroma_wf(name='ica_aroma_wf')

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
    if use_aroma:
        workflow.connect([
            (inputnode,ica_aroma_wf,
                 [('epi_mni', 'inputnode.epi_mni'),
                  ('epi_mask_mni', 'inputnode.epi_mask_mni'),
                  ('movpar_file', 'inputnode.movpar_file')]),
            (ica_aroma_wf,concat,
                [('outputnode.motion_ICs','aroma')]),
            (ica_aroma_wf,outputnode,
                [('outputnode.out_report', 'ica_aroma_report')])
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
                                           tcompcor, acompcor, motion, aroma[0], aroma[1]]
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


def get_ica_confounds(ica_out_dir):
    import os
    import numpy as np

    #pass in numpy array and column base name to generate headers
    def add_header_func(np_arr,col_base):
        import pandas as pd
        from sys import version_info
        PY3 = version_info[0] > 2


        df = pd.DataFrame(np_arr, columns=[str(col_base)+str(x) for index in enumerate(np_arr[0])])
        df.to_csv(str(col_base)+"ica_confounds.tsv", sep="\t" if PY3 else '\t'.encode(), index=None)

        return os.path.abspath(str(col_base)+"ica_confounds.tsv")
    #partial regression (currently inaccurate)
    #TODO fix partial regression
    def calc_residuals(x,y):
        X = np.column_stack(x+[[1]*len(x[0])])
        print(str(y))
        beta_hat = np.linalg.lstsq(X,y)[0]
        y_hat = np.dot(X,beta_hat)
        residuals = y - y_hat

        return residuals

    melodic_mix = os.path.join(ica_out_dir,'melodic.ica/melodic_mix')
    motion_ICs = os.path.join(ica_out_dir,'classified_motion_ICs.txt')

    #-1 since lists start at index 0
    motion_ic_indices = np.loadtxt(motion_ICs,dtype=int,delimiter=',')-1
    melodic_mix_arr = np.loadtxt(melodic_mix,ndmin=2)

    #transpose melodic_mix_arr so indices refer to the correct dimension
    aggr_confounds = np.asarray([melodic_mix_arr.T[x] for x in motion_ic_indices])


    #the "good" ics, (e.g. not motion related)
    good_ic_arr = np.delete(melodic_mix_arr, aggr_confounds, 1).T

    #nonaggr denoising confounds
    nonaggr_confounds = np.asarray([calc_residuals(good_ic_arr,y) for y in aggr_confounds ])

    aggr_tsv = add_header_func(aggr_confounds.T,'aggr_')
    nonaggr_tsv = add_header_func(aggr_confounds.T,'nonaggr_')
    #save the outputs as txt files
    #np.savetxt(aggr_confounds_txt,aggr_confounds.T,fmt=str('%.10f'),delimiter=str('\t'))
    #np.savetxt(nonaggr_confounds_txt,nonaggr_confounds.T,fmt=str('%.10f'),delimiter=str('\t'))
    ic_confounds = (aggr_tsv, nonaggr_tsv)

    return ic_confounds

def init_ica_aroma_wf(name='ica_aroma_wf'):
    #standard mask (assuming epi is in mni space)
    #mni_mask = fsl.Info.standard_image('MNI152_T1_2mm_brain_mask.nii.gz')

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(utility.IdentityInterface(
        fields=['epi_mni', 'movpar_file', 'epi_mask_mni']),
                        name='inputnode')

    outputnode = pe.Node(utility.IdentityInterface(
        fields=['motion_ICs','melodic_report']), name='outputnode')

    #smoothing node (SUSAN)

    #functions to help set SUSAN
    def getbtthresh(medianval):
        return 0.75 * medianval

    def getusans_func(image,thresh):
        return [tuple([image,thresh])]

    calc_median_val = pe.Node(fsl.ImageStats(op_string='-k %s -p 50'),
                              name='calc_median_val')

    calc_epi_mean = pe.Node(fsl.MeanImage(),
                            name='calc_epi_mean')

    brightness_threshold = pe.Node(utility.Function(input_names=['medianval'],
                                            output_names=['thresh'],
                                            function=getbtthresh),
                                        name='brightness_threshold')

    getusans = pe.Node(utility.Function(input_names=['image','thresh'],
                                     output_names=['usans'],
                                     function=getusans_func),
                            name='getusans')

    smooth = pe.Node(fsl.SUSAN(fwhm=6.0),
                          name='smooth')

    #melodic node
    melodic = pe.Node(nws.MELODICRPT(no_bet=True,
                                        no_mask=True,
                                        no_mm=True,
                                        generate_report=True), name="melodic")

    #ica_aroma node
    #TODO change to none once there is agreement between my partial regression
    #and the partial regression completed by fsl_regfilt
    ica_aroma = pe.Node(aroma.ICA_AROMA(denoise_type='both'),
                             name='ica_aroma')

    #set the output directory manually (until pull request #2056 is merged.)
    ica_out_dir = ica_aroma.output_dir()
    ica_aroma.set_input('out_dir',ica_out_dir)



    #extract the confound ICs from the results



    ica_confound = pe.Node(utility.Function(input_names=['ica_out_dir'],
                                    output_names=['ic_confounds'],
                                    function=get_ica_confounds),
                            name='ic_confounds')

    #connect the nodes
    workflow.connect([
        #Connect nodes to complete smoothing
        (inputnode, calc_median_val,
            [('epi_mni','in_file'),
             ('epi_mask_mni', 'mask_file')]),
        (calc_median_val, brightness_threshold,
            [('out_stat','medianval')]),
        (inputnode, calc_epi_mean,
            [('epi_mni','in_file')]),
        (calc_epi_mean, getusans,
            [('out_file','image')]),
        (calc_median_val, getusans,
            [('out_stat','thresh')]),
        (inputnode, smooth,
            [('epi_mni','in_file')]),
        (getusans, smooth,
            [('usans', 'usans')]),
        (brightness_threshold, smooth,
            [('thresh','brightness_threshold')]),
        #connect smooth to melodic
        (smooth, melodic,
            [('smoothed_file','in_files')]),
        #connect nodes to ICA-AROMA
        (smooth, ica_aroma,
            [('smoothed_file', 'in_file')]),
        (inputnode, ica_aroma,
            [('movpar_file','motion_parameters')]),
        (melodic, ica_aroma,
            [('out_dir','melodic_dir')]),
        #geneerate tsvs from ICA_AROMA
        (ica_aroma, ica_confound,
            [('out_dir','ica_out_dir')]),
        #output for processing and reporting
        (ica_confound, outputnode,
            [('ic_confounds','motion_ICs')]),
            #TODO change melodic report to reflect noise and non-noise components
        (melodic, outputnode,
            [('out_report','ica_aroma_report')]),
        ])

    return workflow


# 170531-23:47:46,227 workflow INFO:
# 	 Creating FreeSurfer processing flow.
# 170531-23:47:51,527 workflow INFO:
# 	 Workflow fmriprep_wf settings: ['check', 'execution', 'logging']
# 170531-23:47:51,718 workflow INFO:
# 	 Running in parallel.
# 170531-23:47:51,727 workflow INFO:
# 	 Executing: split ID: 1
# 170531-23:47:51,729 workflow INFO:
# 	 Executing: ds_ica_aroma_report ID: 0
# 170531-23:47:51,732 workflow INFO:
# 	 Executing node ds_ica_aroma_report in dir: /root/src/fmriprep/work/fmriprep_wf/single_subject_controlGE140_wf/func_preproc_task_flanker_wf/func_reports_wf/ds_ica_aroma_report
# 170531-23:47:51,734 workflow INFO:
# 	 Executing node split in dir: /root/src/fmriprep/work/fmriprep_wf/single_subject_controlGE140_wf/func_preproc_task_flanker_wf/epi_hmc_wf/split
# 170531-23:47:51,742 workflow ERROR:
# 	 ['Node ds_ica_aroma_report failed to run on host 42516ca31d70.']
# 170531-23:47:51,747 workflow INFO:
# 	 Running: fslsplit /data/sub-controlGE140/ses-pre/func/sub-controlGE140_task-flanker_bold.nii.gz -t
# 170531-23:47:51,763 workflow INFO:
# 	 Saving crash info to /out/fmriprep/sub-controlGE140/log/20170531-234742_ccd14b1b-23d1-40a7-a7d8-970b6ccd0cba/crash-20170531-234751-root-ds_ica_aroma_report-398d89bc-43c0-49ae-84df-e64276b75d17.pklz
# 170531-23:47:51,764 workflow INFO:
# 	 Traceback (most recent call last):
#   File "/usr/local/miniconda/lib/python3.6/site-packages/nipype/pipeline/plugins/multiproc.py", line 322, in _send_procs_to_workers
#     self.procs[jobid].run()
#   File "/usr/local/miniconda/lib/python3.6/site-packages/nipype/pipeline/engine/nodes.py", line 372, in run
#     self._run_interface()
#   File "/usr/local/miniconda/lib/python3.6/site-packages/nipype/pipeline/engine/nodes.py", line 482, in _run_interface
#     self._result = self._run_command(execute)
#   File "/usr/local/miniconda/lib/python3.6/site-packages/nipype/pipeline/engine/nodes.py", line 613, in _run_command
#     result = self._interface.run()
#   File "/usr/local/miniconda/lib/python3.6/site-packages/nipype/interfaces/base.py", line 1066, in run
#     self._check_mandatory_inputs()
#   File "/usr/local/miniconda/lib/python3.6/site-packages/nipype/interfaces/base.py", line 971, in _check_mandatory_inputs
#     raise ValueError(msg)
# ValueError: DerivativesDataSink requires a value for input 'in_file'. For a list of required inputs, see DerivativesDataSink.help()
#
# 170531-23:47:51,776 workflow INFO:
