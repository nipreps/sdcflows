from __future__ import division

import logging

from nipype.interfaces.io import JSONFileGrabber
from nipype.interfaces import utility as niu
from nipype.interfaces import ants
from nipype.workflows.dmri import fsl
from nipype.pipeline import engine as pe
from nipype.utils import (b0_indices, time_avg, apply_all_corrections, b0_average,
                          hmc_split, dwi_flirt, eddy_rotate_bvecs, rotate_bvecs,
                          insert_mat, extract_bval, recompose_dwi, recompose_xfm,
                          siemens2rads, rads2radsec, demean_image,
                          cleanup_edge_pipeline, add_empty_vol, vsm2warp,
                          compute_readout,)

class PhaseDiffAndMagnitudes(FieldmapDecider):
    ''' Fieldmap preprocessing workflow for fieldmap data structure 
    8.9.1 in BIDS 1.0.0: one phase diff and at least one magnitude image'''

    def __init__():
        return _make_workflow()

    # based on
    # https://github.com/nipy/nipype/blob/bd36a5dadab73e39d8d46b1f1ad826df3fece5c1/nipype/workflows/dmri/fsl/artifacts.py#L514
    def _make_workflow(name='phase_diff_and_magnitudes', interp='Linear',
                       fugue_params=dict(smooth3d=2.0)):
        """
        Maps the B0 field as proposed by [Jezzard95]_. Phase
        unwrapping is performed using `PRELUDE
        <http://fsl.fmrib.ox.ac.uk/fsl/fsl-4.1.9/fugue/prelude.html>`_
        [Jenkinson03]_. Preparation of the fieldmap is performed reproducing the
        script in FSL `fsl_prepare_fieldmap
        <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#SIEMENS_data>`_.
        
        .. warning:: Only SIEMENS format fieldmaps are supported.
        
        .. admonition:: References
    
        .. [Jezzard95] Jezzard P, and Balaban RS, `Correction for geometric
        distortion in echo planar images from B0 field variations
        <http://dx.doi.org/10.1002/mrm.1910340111>`_,
        MRM 34(1):65-73. (1995). doi: 10.1002/mrm.1910340111.
        
        .. [Jenkinson03] Jenkinson M., `Fast, automated, N-dimensional
        phase-unwrapping algorithm <http://dx.doi.org/10.1002/mrm.10354>`_,
        MRM 49(1):193-197, 2003, doi: 10.1002/mrm.10354.
        
        """
        inputnode = pe.Node(niu.IdentityInterface(
            fields=['in_file', 'in_ref', 'in_mask', 'bmap_pha', 'bmap_mag',
                    'settings']), name='inputnode')
        outputnode = pe.Node(niu.IdentityInterface(
            fields=['out_file', 'out_vsm', 'out_warp']), name='outputnode')

        r_params = _make_node_r_params()

        eff_echo = pe.Node(niu.Function(function=_eff_t_echo,
                                        input_names=['echospacing', 'acc_factor'],
                                        output_names=['eff_echo']), name='EffEcho')
        firstmag = pe.Node(fsl.ExtractROI(t_min=0, t_size=1), name='GetFirst')
        n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='Bias')
        bet = pe.Node(fsl.BET(frac=0.4, mask=True), name='BrainExtraction')
        dilate = pe.Node(fsl.maths.MathsCommand(
            nan2zeros=True, args='-kernel sphere 5 -dilM'), name='MskDilate')
        pha2rads = pe.Node(niu.Function(
            input_names=['in_file'], output_names=['out_file'],
            function=siemens2rads), name='PreparePhase')
        prelude = pe.Node(fsl.PRELUDE(process3d=True), name='PhaseUnwrap')
        rad2rsec = pe.Node(niu.Function(
            input_names=['in_file', 'delta_te'], output_names=['out_file'],
            function=rads2radsec), name='ToRadSec')
        baseline = pe.Node(niu.Function(
            input_names=['in_file', 'index'], output_names=['out_file'],
            function=time_avg), name='Baseline')
        
        fmm2b0 = _make_node_fmm2b0()
        
        applyxfm = pe.Node(ants.ApplyTransforms(
            dimension=3, interpolation=interp), name='FMp_to_B0')
        pre_fugue = pe.Node(fsl.FUGUE(save_fmap=True), name='PreliminaryFugue')
        demean = pe.Node(niu.Function(
            input_names=['in_file', 'in_mask'], output_names=['out_file'],
            function=demean_image), name='DemeanFmap')
        
        cleanup = cleanup_edge_pipeline()
        
        addvol = pe.Node(niu.Function(
            input_names=['in_file'], output_names=['out_file'],
            function=add_empty_vol), name='AddEmptyVol')
        vsm = pe.Node(fsl.FUGUE(save_shift=True, **fugue_params),
                      name="ComputeVSM")
        split = pe.Node(fsl.Split(dimension='t'), name='SplitDWIs')
        merge = pe.Node(fsl.Merge(dimension='t'), name='MergeDWIs')

        unwarp = pe.MapNode(fsl.FUGUE(icorr=True, forward_warping=False),
                            iterfield=['in_file'], name='UnwarpDWIs')
        thres = pe.MapNode(fsl.Threshold(thresh=0.0), iterfield=['in_file'],
                           name='RemoveNegative')
        vsm2dfm = _make_workflow_vsm2warp()
        
        wf = pe.Workflow(name=name)
        wf.connect([
            (inputnode, r_params, [('settings', 'in_file')]),
            (r_params, eff_echo, [('echospacing', 'echospacing'),
                                  ('acc_factor', 'acc_factor')]),
            (inputnode, pha2rads, [('bmap_pha', 'in_file')]),
            (inputnode, firstmag, [('bmap_mag', 'in_file')]),
            (inputnode, baseline, [('in_file', 'in_file'),
                                   ('in_ref', 'index')]),
            (firstmag, n4, [('roi_file', 'input_image')]),
            (n4, bet, [('output_image', 'in_file')]),
            (bet, dilate, [('mask_file', 'in_file')]),
            (pha2rads, prelude, [('out_file', 'phase_file')]),
            (n4, prelude, [('output_image', 'magnitude_file')]),
            (dilate, prelude, [('out_file', 'mask_file')]),
            (r_params, rad2rsec, [('delta_te', 'delta_te')]),
            (prelude, rad2rsec, [('unwrapped_phase_file', 'in_file')]),
            
            (baseline, fmm2b0, [('out_file', 'fixed_image')]),
            (n4, fmm2b0, [('output_image', 'moving_image')]),
            (inputnode, fmm2b0, [('in_mask', 'fixed_image_mask')]),
            (dilate, fmm2b0, [('out_file', 'moving_image_mask')]),
            
            (baseline, applyxfm, [('out_file', 'reference_image')]),
            (rad2rsec, applyxfm, [('out_file', 'input_image')]),
            (fmm2b0, applyxfm, [
                ('forward_transforms', 'transforms'),
                ('forward_invert_flags', 'invert_transform_flags')]),
            
            (applyxfm, pre_fugue, [('output_image', 'fmap_in_file')]),
            (inputnode, pre_fugue, [('in_mask', 'mask_file')]),
            (pre_fugue, demean, [('fmap_out_file', 'in_file')]),
            (inputnode, demean, [('in_mask', 'in_mask')]),
            (demean, cleanup, [('out_file', 'inputnode.in_file')]),
            (inputnode, cleanup, [('in_mask', 'inputnode.in_mask')]),
            (cleanup, addvol, [('outputnode.out_file', 'in_file')]),
            (inputnode, vsm, [('in_mask', 'mask_file')]),
            (addvol, vsm, [('out_file', 'fmap_in_file')]),
            (r_params, vsm, [('delta_te', 'asym_se_time')]),
            (eff_echo, vsm, [('eff_echo', 'dwell_time')]),
            (inputnode, split, [('in_file', 'in_file')]),
            (split, unwarp, [('out_files', 'in_file')]),
            (vsm, unwarp, [('shift_out_file', 'shift_in_file')]),
            (r_params, unwarp, [
                (('enc_dir', _fix_enc_dir), 'unwarp_direction')]),
            (unwarp, thres, [('unwarped_file', 'in_file')]),
            (thres, merge, [('out_file', 'in_files')]),
            (r_params, vsm2dfm, [
                (('enc_dir', _fix_enc_dir), 'inputnode.enc_dir')]),
            (merge, vsm2dfm, [('merged_file', 'inputnode.in_ref')]),
            (vsm, vsm2dfm, [('shift_out_file', 'inputnode.in_vsm')]),
            (merge, outputnode, [('merged_file', 'out_file')]),
            (vsm, outputnode, [('shift_out_file', 'out_vsm')]),
            (vsm2dfm, outputnode, [('outputnode.out_warp', 'out_warp')])
        ])
        return wf

    def _make_node_r_params():
        epi_defaults = {'delta_te': 2.46e-3, 'echospacing': 0.77e-3,
                        'acc_factor': 2, 'enc_dir': u'AP'}        
        return pe.Node(JSONFileGrabber(defaults=epi_defaults),
                           name='SettingsGrabber')

    def _make_node_fmm2b0():
        fmm2b0 = pe.Node(ants.Registration(output_warped_image=True),
                         name="FMm_to_B0")

        fmm2b0.inputs.transforms = ['Rigid'] * 2
        fmm2b0.inputs.transform_parameters = [(1.0,)] * 2
        fmm2b0.inputs.number_of_iterations = [[50], [20]]
        fmm2b0.inputs.dimension = 3
        fmm2b0.inputs.metric = ['Mattes', 'Mattes']
        fmm2b0.inputs.metric_weight = [1.0] * 2
        fmm2b0.inputs.radius_or_number_of_bins = [64, 64]
        fmm2b0.inputs.sampling_strategy = ['Regular', 'Random']
        fmm2b0.inputs.sampling_percentage = [None, 0.2]
        fmm2b0.inputs.convergence_threshold = [1.e-5, 1.e-8]
        fmm2b0.inputs.convergence_window_size = [20, 10]
        fmm2b0.inputs.smoothing_sigmas = [[6.0], [2.0]]
        fmm2b0.inputs.sigma_units = ['vox'] * 2
        fmm2b0.inputs.shrink_factors = [[6], [1]]  # ,[1] ]
        fmm2b0.inputs.use_estimate_learning_rate_once = [True] * 2
        fmm2b0.inputs.use_histogram_matching = [True] * 2
        fmm2b0.inputs.initial_moving_transform_com = 0
        fmm2b0.inputs.collapse_output_transforms = True
        fmm2b0.inputs.winsorize_upper_quantile = 0.995

        return fmm2b0

    def _make_workflow_vsm2warp():
        vsm2dfm = vsm2warp()
        vsm2dfm.inputs.inputnode.scaling = 1.0
        return vsm2dfm
