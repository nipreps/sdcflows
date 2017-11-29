# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Creating a T2*-map with mutli-echo BOLD data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: init_bold_t2s_map_wf

"""
from niworkflows.nipype import logging
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import (utility as niu, afni)
from niworkflows.interfaces.utils import CopyXForm

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('workflow')

def init_bold_t2s_map_wf(metadata, name='bold_t2s_map_wf'):
    """

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from fmriprep.workflows.bold import init_bold_t2s_map_wf
        wf = init_bold_t2s_map_wf(
            metadata={"RepetitionTime": 2.0)

    **Parameters**

        metadata : dict
            BIDS metadata for BOLD file
        name : str
            Name of workflow (default: ``bold_t2s_map_wf``)

    **Inputs**

        bold_file
            BOLD series NIfTI file

    **Outputs**

        t2s_map
            T2*-map
    """
    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=['bold_file']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['t2s_map']), name='outputnode')

    LOGGER.log(25, 'Generating a T2*-map.')
    pass
