#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility workflows
"""
from __future__ import print_function, division, absolute_import, unicode_literals

from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import ants
from fmriprep.interfaces import CopyHeader

LOGGER = logging.getLogger('workflow')


def init_n4bias_wf(name='n4bias_wf'):
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['in_file']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_file']), name='outputnode')

    n4_correct = pe.Node(ants.N4BiasFieldCorrection(dimension=3), name='n4_correct')
    orig_hdr = pe.Node(CopyHeader(), name='orig_hdr')

    workflow.connect([
        (inputnode, n4_correct, [('in_file', 'input_image')]),
        (inputnode, orig_hdr, [('in_file', 'hdr_file')]),
        (n4_correct, orig_hdr, [('output_image', 'in_file')]),
        (orig_hdr, outputnode, [('out_file', 'out_file')]),
        ])

    return workflow
