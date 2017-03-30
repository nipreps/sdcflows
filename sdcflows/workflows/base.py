#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base fieldmap estimation
------------------------

* Subject can have no data for fieldmap estimation - unwarping should not modify
  the images.
* Subject can have phase-difference data.
* Subject can have a fieldmap acquisition.
* Subject can have :abbr:`pepolar (Phase-Encoding POLArity)` image series for a
  phase-encoding based estimation.
* Subject can have two or three of the above - return average.


"""
from __future__ import print_function, division, absolute_import, unicode_literals

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from nipype import logging

LOGGER = logging.getLogger('workflow')


def fmap_estimator(subject_data, settings=None):
    """
    This workflow selects the fieldmap estimation data available for the subject and
    returns the estimated fieldmap in mm, along with a corresponding reference image.
    Current implementation applies the first type of fieldmap estimation found in the
    following order of precedence:

      * "Natural" fieldmaps
      * Phase-difference fieldmaps
      * PEPolar fieldmaps.


    .. topic:: Outputs

        fmap
          The estimated fieldmap itself IN UNITS OF Hz.
        fmap_ref
          the anatomical reference for the fieldmap (magnitude image, corrected SEm, etc.)
        fmap_mask
          a brain mask for the fieldmap


    """
    if not subject_data.get('fmap', []):
        LOGGER.info('Fieldmap: no data found for estimation')
        # When there is no data for fieldmap estimation, just return None
        return None

    # Otherwise, build the appropriate workflow(s)
    workflow = pe.Workflow(name='FieldmapEstimation')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap', 'fmap_ref', 'fmap_mask']), name='outputnode')

    estimator_wf = None
    if any(['fieldmap' in fname for fname in subject_data['fmap']]):
        from .fmap import fmap_workflow
        LOGGER.info('Fieldmap estimation: fieldmap images found')
        fmapwf = fmap_workflow(settings=settings)
        # set inputs
        fmapwf.inputs.inputnode.input_images = subject_data['fmap']
        estimator_wf = fmapwf

    elif any(['phase' in fname for fname in subject_data['fmap']]):
        from .phdiff import phdiff_workflow
        LOGGER.info('Fieldmap estimation: phase-difference images found')
        phwf = phdiff_workflow(settings=settings)
        # set inputs
        phwf.inputs.inputnode.input_images = subject_data['fmap']
        estimator_wf = phwf

    elif any(['epi' in fname for fname in subject_data['fmap']]):
        from .pepolar import pepolar_workflow
        LOGGER.info('Fieldmap estimation: phase-encoding images found')
        pewf = pepolar_workflow(settings=settings)
        # set inputs
        pewf.inputs.inputnode.input_images = subject_data['fmap'] + subject_data['sbref']
        estimator_wf = pewf

    if estimator_wf is None:
        LOGGER.warn(
            'Fieldmap data found, but no estimation workflow could be built for it. '
            'Fieldmap data available = [%s]', ', '.join(subject_data['fmap']))
        return None

    workflow.connect([
        (estimator_wf, outputnode, [
            ('outputnode.fmap', 'fmap'),
            ('outputnode.fmap_ref', 'fmap_ref'),
            ('outputnode.fmap_mask', 'fmap_mask')])
    ])

    return workflow
