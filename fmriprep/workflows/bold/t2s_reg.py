# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
T2*-map registration workflow for multi-echo BOLD data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_t2s_reg_wf

"""
import numpy as np
import nibabel as nib

from niworkflows.nipype import logging
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import (utility as niu, afni)
from niworkflows.interfaces.utils import CopyXForm

from ...interfaces.multiecho import T2SMap

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('workflow')


def init_bold_t2s_reg_wf(name='bold_t2s_reg_wf', use_compression=True):
    """

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold import init_bold_t2s_map_wf
        wf = init_bold_t2s_map_wf(
            metadata={"RepetitionTime": 2.0)

    **Parameters**

        name : str
            Name of workflow (default: ``bold_t2s_map_wf``)
        use_compression : bool
            Save registered BOLD series as ``.nii.gz``

    **Inputs**

        in_file
            Reference BOLD image to be registered
        t1_brain
            Skull-stripped T1-weighted structural image
        t1_seg
            FAST segmentation of ``t1_brain``

    **Outputs**

        t2s_map
            T2*-map
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file', 'tes']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['t2s_map']), name='outputnode')

    # inputnode = pe.Node(niu.IdentityInterface(fields=['t2svol', 'anat']),
    #                     name='inputnode')
    #
    # outputnode = pe.Node(niu.IdentityInterface(fields=['coreg_params']),
    #                      name='outputnode')

    LOGGER.log(25, 'Generating a T2*-map, using in EPI-T1 coregistration.')

    get_thr = pe.Node(fsl.ImageStats(op_string='-P 50'),
                      name='get_thr')

    fmt_expr = pe.Node(name='fmt_expr',
                       interface=Function(input_names=['val'],
                                          output_names=['expr_string'],
                                          function=_format_expr))

    apply_thr = pe.Node(afni.Calc(), name='apply_thr')

    t1_seg = pe.Node(fsl.FAST(use_priors=True,
                              probability_maps=True), name='t1_seg')

    align = pe.Node(afni.Allineate(out_file='mepi_al.nii.gz',
                                   out_matrix='mepi_al_mat.1D',
                                   source_automask=2,
                                   warp_type='affine_general',
                                   args='-weight_frac 1.0 -lpc'
                                        '-maxshf 30 -maxrot 30'
                                        '-maxscl 1.01'),
                    name='align')

    workflow.connect([
                    (inputnode, get_thr, [('t2svol', 'in_file')]),
                    (inputnode, align, [('anat', 'in_file'),
                                        ('anat', 'master')]),
                    (get_thr, fmt_expr, [('out_stat', 'val')]),
                    (inputnode, apply_thr, [('t2svol', 'in_file_a')]),
                    (fmt_expr, apply_thr, [('expr_string', 'expr')]),
                    (apply_thr, t1_seg, [('out_file', 'in_files')]),
                    (apply_thr, align, [('out_file', 'reference')]),
                    (t1_seg, align, [('tissue_class_map', 'weight_file')]),
                    (align, outputnode, [('matrix', 'coreg_params')])
                    ])

    workflow.write_graph(graph2use='colored', simple_form=True)

    return workflow


def _format_expr(val):
    """
    Generates string for use as `expr`
    input in afni.Calc()

    Parameters
    ----------
    val: float
        Threshold generated from fsl.ImageStats()

    Outputs
    ----------
    expr_string
        Expression to be applyed with afni.Calc()
    """
    expr_string = 'a*isnegative(a-2*{})'.format(val)
    return expr_string
