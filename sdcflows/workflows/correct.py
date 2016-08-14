#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Apply susceptibility distortion correction (SDC)

"""
import pkg_resources as pkgr

from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nipype.interfaces import ants
from nipype.interfaces import io as nio
from nipype.interfaces import utility as niu

from fmriprep.utils.misc import gen_list
from fmriprep.interfaces import ReadSidecarJSON
from fmriprep.workflows.fieldmap.base import create_encoding_file

SDC_CORRECT_NAME = 'SDC_Apply'

def sdc_correct(name=SDC_CORRECT_NAME, ref_vol=None):
    """
    This workflow takes an estimated fieldmap and a target image and applies TOPUP,
    an :abbr:`SDC (susceptibility-derived distortion correction)` method in FSL to
    unwarp the target image.

    Input fields:
    ~~~~~~~~~~~~~

      inputnode.in_file - the image(s) to which this correction will be applied
      inputnode.in_mask - a brain mask corresponding to the in_file image
      inputnode.fmap_ref - the fieldmap reference (generally, a *magnitude* image or the
                           resulting SE image)
      inputnode.fmap_mask - a brain mask in fieldmap-space
      inputnode.fieldmap - a fieldmap in Hz
      inputnode.hmc_movpar - the head motion parameters (iff inputnode.in_file is only
                             one 4D file)

    Output fields:
    ~~~~~~~~~~~~~~

      outputnode.out_file - the in_file after susceptibility-distortion correction.

    """

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_file', 'in_mask', 'fmap_ref', 'fmap_mask', 'fieldmap',
                'hmc_movpar']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    # Compute movpar file iff we have several images with different
    # PE directions.
    align = pe.Node(niu.Function(
        input_names=['in_files', 'in_movpar', 'in_ref'],
        output_names=['out_file', 'out_movpar', 'method', 'ref_vol'],
        function=_multiple_pe_hmc), name='AlignMultiplePE')
    align.inputs.in_ref = ref_vol

    # Read metadata
    meta = pe.MapNode(ReadSidecarJSON(fields=['TotalReadoutTime', 'PhaseEncodingDirection']),
                      iterfield=['in_file'], name='metadata')

    encfile = pe.Node(interface=niu.Function(
        input_names=['input_images', 'in_dict'], output_names=['parameters_file'],
        function=create_encoding_file), name='TopUp_encfile', updatehash=True)

    fslsplit = pe.Node(fsl.Split(dimension='t'), name='ImageHMCSplit')

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    fmap2ref = pe.Node(ants.Registration(output_warped_image=True),
                       name='Fieldmap2ImageRegistration')

    grabber = nio.JSONFileGrabber()
    setattr(grabber, '_always_run', False)
    fmap2ref_params = pe.Node(grabber, name='Fieldmap2ImageRegistration_params')
    fmap2ref_params.inputs.in_file = (
        pkgr.resource_filename('fmriprep', 'data/fmap-any_registration.json'))

    applyxfm = pe.Node(ants.ApplyTransforms(
        dimension=3, interpolation='Linear'), name='Fieldmap2ImageApply')

    to_coeff = pe.Node(niu.Function(
        input_names=['in_file', 'in_ref'], output_names=['out_file'],
        function=_gen_coeff), name='GenCoeffFile')

    # Use the least-squares method to correct the dropout of the SBRef images
    unwarp = pe.Node(fsl.ApplyTOPUP(), name='TopUpApply')


    if ref_vol is None:
        getref = pe.Node(fsl.MeanImage(dimension='T'), name='ImageAverage')

        workflow.connect([
            (align, getref, [('out_file', 'in_file')]),
            (getref, fmap2ref, [('out_file', 'fixed_image')]),
            (getref, applyxfm, [('out_file', 'reference_image')]),
            (getref, to_coeff, [('out_file', 'in_ref')]),
        ])
    else:
        getref = pe.Node(niu.Select(), name='PickFirst')
        workflow.connect([
            (fslsplit, getref, [('out_files', 'inlist')]),
            (align, getref, [('ref_vol', 'index')]),
            (getref, fmap2ref, [('out', 'fixed_image')]),
            (getref, applyxfm, [('out', 'reference_image')]),
            (getref, to_coeff, [('out', 'in_ref')]),
        ])


    workflow.connect([
        (inputnode, meta, [('in_file', 'in_file')]),
        (inputnode, align, [('in_file', 'in_files'),
                            ('hmc_movpar', 'in_movpar')]),
        (inputnode, applyxfm, [('fieldmap', 'input_image')]),
        (inputnode, encfile, [('in_file', 'input_images')]),
        (meta, encfile, [('out_dict', 'in_dict')]),

        (align, unwarp, [('out_movpar', 'in_topup_movpar'),
                         ('method', 'method')]),

        (inputnode, fmap2ref, [('in_mask', 'fixed_image_mask'),
                               ('fmap_ref', 'moving_image'),
                               ('fmap_mask', 'moving_image_mask')]),
        (fmap2ref, applyxfm, [
            ('forward_transforms', 'transforms'),
            ('forward_invert_flags', 'invert_transform_flags')]),
        (align, fslsplit, [('out_file', 'in_file')]),
        (applyxfm, to_coeff, [('output_image', 'in_file')]),
        (fslsplit, unwarp, [('out_files', 'in_files'),
                            (('out_files', gen_list), 'in_index')]),
        (to_coeff, unwarp, [('out_file', 'in_topup_fieldcoef')]),
        (encfile, unwarp, [('parameters_file', 'encoding_file')]),
        (unwarp, outputnode, [('out_corrected', 'out_file')])
    ])

    # Connect registration settings in the end, not to clutter the code
    workflow.connect([
        (fmap2ref_params, fmap2ref, [
            ('transforms', 'transforms'),
            ('transform_parameters', 'transform_parameters'),
            ('number_of_iterations', 'number_of_iterations'),
            ('dimension', 'dimension'),
            ('metric', 'metric'),
            ('metric_weight', 'metric_weight'),
            ('radius_or_number_of_bins', 'radius_or_number_of_bins'),
            ('sampling_strategy', 'sampling_strategy'),
            ('sampling_percentage', 'sampling_percentage'),
            ('convergence_threshold', 'convergence_threshold'),
            ('convergence_window_size', 'convergence_window_size'),
            ('smoothing_sigmas', 'smoothing_sigmas'),
            ('sigma_units', 'sigma_units'),
            ('shrink_factors', 'shrink_factors'),
            ('use_estimate_learning_rate_once', 'use_estimate_learning_rate_once'),
            ('use_histogram_matching', 'use_histogram_matching'),
            ('initial_moving_transform_com', 'initial_moving_transform_com'),
            ('collapse_output_transforms', 'collapse_output_transforms'),
            ('winsorize_upper_quantile', 'winsorize_upper_quantile'),
            ('winsorize_lower_quantile', 'winsorize_lower_quantile')
        ])
    ])

    return workflow


def _multiple_pe_hmc(in_files, reference, in_movpar, in_ref):
    """
    This function interprets that we are dealing with a
    multiple PE (phase encoding) input if it finds several
    files in in_files.

    If we have several images with various PE directions,
    it will compute the HMC parameters between them using
    an embedded workflow.

    It just forwards the two inputs otherwise.
    """
    if len(in_files) == 1:
        return in_files[0], in_movpar, 'jac', in_ref
    else:
        import os
        from nipype.interfaces import fsl
        from fmripep.interfaces import FormatHMCParam

        if in_ref is None:
            in_ref = 0

        # Head motion correction
        fslmerge = fsl.Merge(dimension='t')
        fslmerge.inputs.in_file = in_files
        hmc = fsl.MCFLIRT(cost='normcorr', ref_vol=in_ref, save_mats=True)
        hmc.inputs.in_file = fslmerge.run().outputs.merged_file
        hmc_res = hmc.run()

        translations = []
        rot_angles = []
        for mat_file in hmc_res.outputs.mat_file:
            avscale = fsl.utils.AvScale(
                all_param=True, mat_file=mat_file, ref_file=in_files[0])
            avres = avscale.run()
            translations.append(avres.outputs.translations)
            rot_angles.append(avres.outputs.rot_angles)

        avs_format = FormatHMCParam(translations=translations,
                                    rot_angles=rot_angles,
                                    fmt='movpar_file')
        return (hmc_res.outputs.out_file,
                avs_format.run().outputs.out_file,
                'lsr', in_ref)

def _gen_coeff(in_file, in_ref, out_file=None):
    """Convert to a valid fieldcoeff"""
    import numpy as np
    import nibabel as nb
    from nipype.interfaces import fsl

    def _gen_fname(in_file, suffix='process'):
        import os.path as op
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        return op.abspath('./{}_{}'.format(fname, suffix))

    if out_file is None:
        out_file = _gen_fname(in_file, 'fieldcoeff.nii.gz')

    # 1. Add one dimension (4D image) of 3D coordinates
    im0 = nb.load(in_file)
    data = np.zeros_like(im0.get_data())
    sizes = data.shape[:3]
    spacings = im0.get_header().get_zooms()[:3]
    im1 = nb.Nifti1Image(data, im0.get_affine(), im0.get_header())
    im4d = nb.concat_images([im0, im1, im1])
    im4d_fname = _gen_fname(in_file, 'field4D.nii.gz')
    im4d.to_filename(im4d_fname)

    # 2. Warputils to compute bspline coefficients
    to_coeff = fsl.WarpUtils(out_format='spline', knot_space=(2, 2, 2))
    to_coeff.inputs.in_file = im4d_fname
    to_coeff.inputs.reference = in_ref

    # 3. Remove unnecessary dims (Y and Z)
    get_first = fsl.ExtractROI(t_min=0, t_size=1)
    get_first.inputs.in_file = to_coeff.run().outputs.out_file

    # 4. Set correct header
    # see https://github.com/poldracklab/preprocessing-workflow/issues/92
    img = nb.load(get_first.run().outputs.roi_file)
    hdr = img.get_header().copy()
    hdr['intent_p1'] = spacings[0]
    hdr['intent_p2'] = spacings[1]
    hdr['intent_p3'] = spacings[2]
    hdr['intent_code'] = 2016

    sform = np.eye(4)
    sform[:3, 3] = sizes
    hdr.set_sform(sform, code='scanner')
    hdr['qform_code'] = 1

    nb.Nifti1Image(img.get_data(), None, hdr).to_filename(out_file)
    return out_file


