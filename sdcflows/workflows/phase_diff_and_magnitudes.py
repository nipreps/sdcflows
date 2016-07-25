from __future__ import division

import logging

from nipype.interfaces.io import JSONFileGrabber
from nipype.interfaces import utility as niu
from nipype.interfaces import ants
from nipype.interfaces import fsl
from nipype.pipeline import engine as pe
from nipype.workflows.dmri.fsl.utils import (siemens2rads, demean_image, cleanup_edge_pipeline,
                                             add_empty_vol)

from fmriprep.interfaces.bids import ReadSidecarJSON
from fmriprep.utils.misc import fieldmap_suffixes

''' Fieldmap preprocessing workflow for fieldmap data structure
8.9.1 in BIDS 1.0.0: one phase diff and at least one magnitude image'''

def _sort_fmaps(fieldmaps):
    ''' just a little data massaging'''
    from fmriprep.workflows.fieldmap.helper import sort_fmaps

    fmaps = sort_fmaps(fieldmaps)
    # there is only one phasediff image
    # there may be more than one magnitude image, but the workflow (GetFirst
    # node) needs to change first

    print(fmaps)
    return fmaps['phasediff'][0], fmaps['magnitude']


def phase_diff_and_magnitudes(name='phase_diff_and_magnitudes', interp='Linear',
                              fugue_params=dict(smooth3d=2.0)):
    """
    Estimates the fieldmap using a phase-difference image and one or more
    magnitude images corresponding to two or more :abbr:`GRE (Gradient Echo sequence)`
    acquisitions. The `original code was taken from nipype
    <https://github.com/nipy/nipype/blob/master/nipype/workflows/dmri/fsl/artifacts.py#L514>`_.

    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['fieldmaps']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['mag_brain',
                                                       'fmap_mask',
                                                       'fmap_fieldcoef', # in sepair this comes from topup.out_fieldcoef; ask what to do
                                                       'fmap_movpar']), # same as above; topup.out_movpar
                         name='outputnode')

    sortfmaps = pe.Node(niu.Function(function=_sort_fmaps,
                                      input_names=['fieldmaps'],
                                      output_names=['phasediff', 'magnitude']),
                        name='SortFmaps')

    # Read phasediff echo times
    meta = pe.Node(ReadSidecarJSON(fields=['EchoTime1', 'EchoTime2']), name='metadata')
    dte = pe.Node(niu.Function(input_names=['in_values'], output_names=['delta_te'],
                               function=_delta_te), name='ComputeDeltaTE')

    # ideally use mcflirt first to align the magnitude images
    magmrg = pe.Node(fsl.Merge(dimension='t'), name='MagsMerge')
    magavg = pe.Node(fsl.MeanImage(dimension='T', nan2zeros=True), name='MagsAverage')

    # de-gradient the fields ("bias/illumination artifact")
    n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='Bias')

    bet = pe.Node(fsl.BET(frac=0.4, mask=True), name='BrainExtraction')
    # uses mask from bet; outputs a mask
    dilate = pe.Node(fsl.maths.MathsCommand(
        nan2zeros=True, args='-kernel sphere 5 -dilM'), name='MskDilate')

    # phase diff -> radians
    pha2rads = pe.Node(niu.Function(
        input_names=['in_file'], output_names=['out_file'],
        function=siemens2rads), name='PreparePhase')

    # FSL PRELUDE will perform phase-unwrapping
    prelude = pe.Node(fsl.PRELUDE(process3d=True), name='PhaseUnwrap')

    rad2rsec = pe.Node(niu.Function(
        input_names=['in_file', 'delta_te'], output_names=['out_file'],
        function=rads2radsec), name='ToRadSec')
    # pre_fugue = pe.Node(fsl.FUGUE(save_fmap=True), name='PreliminaryFugue')

    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, sortfmaps, [('fieldmaps', 'fieldmaps')]),
        (sortfmaps, meta, [('phasediff', 'in_file')]),
        (sortfmaps, magmrg, [('magnitude', 'in_files')]),
        (magmrg, magavg, [('merged_file', 'in_file')]),
        (magavg, n4, [('out_file', 'input_image')]),
        (n4, prelude, [('output_image', 'magnitude_file')]),
        (n4, bet, [('output_image', 'in_file')]),
        (bet, dilate, [('mask_file', 'in_file')]),
        (dilate, prelude, [('out_file', 'mask_file')]),
        (sortfmaps, pha2rads, [('phasediff', 'in_file')]),
        (pha2rads, prelude, [('out_file', 'phase_file')]),
        (meta, dte, [('out_dict', 'in_values')]),
        (dte, rad2rsec, [('delta_te', 'delta_te')]),
        (prelude, rad2rsec, [('unwrapped_phase_file', 'in_file')]),
        (rad2rsec, outputnode, [('out_file', 'fmap_fieldcoef')]),
        (bet, outputnode, [('mask_file', 'fmap_mask'),
                           ('out_file', 'mag_brain')])
    ])


    # ingest_fmap_data = _ingest_fieldmap_data_workflow()
    # wrangle_fmap_data = _wrangle_fieldmap_data_workflow()
    # vsm = pe.Node(fsl.FUGUE(save_shift=True, **fugue_params),
    #               name="ComputeVSM")
    # vsm2topup = pe.Node(niu.Function(function=_vsm_to_topup,
    #                                  input_names=['vsm'], # voxel shift map file
    #                                  output_names=['fmap_fieldcoef', 'fmap_movpar']),
    #                     name='vsm2topup')

    # workflow = pe.Workflow(name=name)
    # workflow.connect([

    #     (sortfmaps, ingest_fmap_data, [('phasediff', 'inputnode.bmap_pha'),
    #                                    ('magnitude', 'inputnode.bmap_mag')]),
    #     (ingest_fmap_data, vsm, [('outputnode.skull_strip_mask_file', 'mask_file')]),
    #     (ingest_fmap_data, outputnode, [('outputnode.mag_brain', 'mag_brain'), # ??? verify
    #                                      ('outputnode.skull_strip_mask_file', 'fmap_mask')]), # ??? verify
    #     (ingest_fmap_data, wrangle_fmap_data, [('outputnode.skull_strip_mask_file',
    #                                             'inputnode.in_mask')]), # ??? verify
    #     (ingest_fmap_data, rad2rsec, [('outputnode.unwrapped_phase_file', 'in_file')]),
    #     (rad2rsec, pre_fugue, [('out_file','fmap_in_file')]), # ??? verify

    #     (ingest_fmap_data, pre_fugue, [('outputnode.skull_strip_mask_file', # ??? verify
    #                                     'mask_file')]), # ??? verify
    #     (pre_fugue, wrangle_fmap_data, [('fmap_out_file',
    #                                      'inputnode.fmap_out_file')]),
    #     (wrangle_fmap_data, vsm, [('outputnode.out_file', 'fmap_in_file')]),

    #     (r_params, rad2rsec, [('delta_te', 'delta_te')]),
    #     (r_params, vsm, [('delta_te', 'asym_se_time')]),
    #     (r_params, eff_echo, [('echospacing', 'echospacing'),
    #                           ('acc_factor', 'acc_factor')]),
    #     (eff_echo, vsm, [('eff_echo', 'dwell_time')]),

    #     (vsm, vsm2topup, [('shift_out_file', 'vsm')]),
    #     (vsm2topup, outputnode, [('fmap_fieldcoef', 'fmap_fieldcoef'),
    #                              ('fmap_movpar', 'fmap_movpar')])
    # ])
    return workflow

# def _make_node_r_params():
#     # find these values in data--see bids spec
#     # delta_te from phase diff
#     # echospacing, acc_factor, enc_dir from sbref/epi
#     epi_defaults = {'delta_te': 2.46e-3, 'echospacing': 0.77e-3,
#                     'acc_factor': 2, 'enc_dir': u'AP'}
#     return pe.Node(JSONFileGrabber(defaults=epi_defaults),
#                    name='SettingsGrabber')

# def _ingest_fieldmap_data_workflow():
#     ''' Takes phase diff and one magnitude file, handles the
#     illumination problem, extracts the brain, and does phase
#     unwrapping usig FSL's PRELUDE'''
#     name="IngestFmapData"

#     inputnode = pe.Node(niu.IdentityInterface(fields=['bmap_mag',
#                                                       'bmap_pha']),
#                         name='inputnode')
#     outputnode = pe.Node(niu.IdentityInterface(
#         fields=['unwrapped_phase_file',
#                 'skull_strip_mask_file',
#                 'mag_brain']),
#                          name='outputnode')


#     return workflow

# def _wrangle_fieldmap_data_workflow():
#     '''  Normalizes ("demeans") fieldmap data, cleans it up and
#     organizes it for output'''
#     name='WrangleFmapData'

#     inputnode = pe.Node(niu.IdentityInterface(fields=['fmap_out_file',
#                                                       'in_mask']),
#                         name='inputnode')
#     outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']),
#                          name='outputnode')

#     demean = pe.Node(niu.Function(
#         input_names=['in_file', 'in_mask'], output_names=['out_file'],
#         function=demean_image), name='DemeanFmap')

#     cleanup = cleanup_edge_pipeline()

#     addvol = pe.Node(niu.Function(
#         input_names=['in_file'], output_names=['out_file'],
#         function=add_empty_vol), name='AddEmptyVol')

#     workflow = pe.Workflow(name=name)
#     workflow.connect([
#         (inputnode, demean, [('fmap_out_file', 'in_file'),
#                              ('in_mask', 'in_mask')]),
#         (demean, cleanup, [('out_file', 'inputnode.in_file')]),

#         (inputnode, cleanup, [('in_mask', 'inputnode.in_mask')]),
#         (cleanup, addvol, [('outputnode.out_file', 'in_file')]),
#         (addvol, outputnode, [('out_file', 'out_file')]),
#     ])
#     return workflow

# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------

def rads2radsec(in_file, delta_te, out_file=None):
    """
    Converts input phase difference map to rads
    """
    import numpy as np
    import nibabel as nb
    import os.path as op
    import math

    if out_file is None:
        fname, fext = op.splitext(op.basename(in_file))
        if fext == '.gz':
            fname, _ = op.splitext(fname)
        out_file = op.abspath('./%s_radsec.nii.gz' % fname)

    im = nb.load(in_file)
    data = im.get_data().astype(np.float32) * (1.0 / delta_te)
    nb.Nifti1Image(data, im.affine, im.header).to_filename(out_file)
    return out_file


def _delta_te(in_values):
    if isinstance(in_values, float):
        te2 = in_values
        te1 = 0.

    if isinstance(in_values, dict):
        te1 = in_values['EchoTime1']
        te2 = in_values['EchoTime2']

    if isinstance(in_values, list):
        te2, te1 = in_values
        if isinstance(te1, list):
            te1 = te1[1]
        if isinstance(te2, list):
            te2 = te2[1]

    return abs(float(te2)-float(te1))

# def _vsm_to_topup(vsm):
#     # OSCAR'S CODE HERE
#     raise NotImplementedError()
#     return fmap_fieldcoef, fmap_movpar

# def _eff_t_echo(echospacing, acc_factor):
#     eff_echo = echospacing / (1.0 * acc_factor)
#     return eff_echo
