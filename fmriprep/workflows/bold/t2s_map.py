# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Creating a T2*-map with mutli-echo BOLD data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: create_last_echo_mask
.. autofunction:: init_bold_t2s_map_wf

"""
import numpy as np
import nibabel as nib

from niworkflows.nipype import logging
from niworkflows.nipype.pipeline import engine as pe
from niworkflows.nipype.interfaces import (utility as niu, afni)
from niworkflows.interfaces.utils import CopyXForm

DEFAULT_MEMORY_MIN_GB = 0.01
LOGGER = logging.getLogger('workflow')


def create_last_echo_mask(echo_list):
    """
    Make a map of longest echo that a voxel can be sampled with,
    with minimum value of map as X value of voxel that has median
    value in the 1st echo. N.B. larger factor leads to bias to lower TEs

    **Inputs**
        echo_list
            List of file names for all echos

    **Outputs**
        last_echo_mask
            numpy array whose values correspond to which
            echo a voxel can last be sampled with
    """
    # First, load each echo and average over time
    echos = []
    for e in echo_list:
        echos.append(np.mean(nib.load(e).get_data(), axis = -1))

    # In the first echo, find the 33rd percentile and the voxel(s)
    # whose average activity is equal to that value
    med_val = np.percentile(echos[0][np.nonzero(echos[0])],
                            33, interpolation="higher")
    vox = echos[0] == med_val

    # For each (averaged) echo, extract the max signal in the
    # identified voxels and divide it by 3-- save as a threshold
    thrs = np.empty(len(echos))
    for echo in echos:
        np.append(thrs, np.max(echo[vox]) / 3)

    # Let's stack the arrays to make this next bit easier
    echo_means = np.stack(echos, axis=-1)

    # Now, we want to find all voxels (in each echo) that show
    # absolute signal greater than our echo-specific threshold
    mthr = np.ones(echo_means.shape)
    for i in range(echo_means.shape[-1]):
        mthr[:, :, :, i] *= thrs[i]
    voxels = np.abs(echo_means) > mthr

    # We save those voxel indices out to an array
    last_echo_mask = np.array(voxels, dtype=np.int).sum(-1)

    return last_echo_mask


def create_t2s_map(echo_list, last_echo_mask, tes):
    """
    **Inputs**
        echo_list
            List of file names for all echos
        last_echo_mask
            numpy array where voxel values correspond to which
            echo a voxel can last be sampled with
        tes

    **Outputs**
        t2s_map
    """
    echos = []
    for e in echo_list:
        echos.append(nib.load(e).get_data())
    echo_data = np.stack(echos, axis=-2)
    nx, ny, nz, Ne, nt = echo_data.shape
    N = nx * ny * nz

    Nm = echo_data[np.nonzero(echo_data)].shape[0]

    t2ss = np.zeros([nx, ny, nz, Ne-1])
    s0vs = t2ss.copy()

    for ne in range(2, Ne + 1):

        #Do Log Linear fit
        B = np.reshape(np.abs(echo_data[:, :ne])+1, (Nm, ne * nt)).transpose()
        B = np.log(B)
        neg_tes = [-1 * te for te in tes[:ne]]
        x = np.array([np.ones(ne), neg_tes])
        X = np.tile(x, (1, nt))
        X = np.sort(X)[:, ::-1].transpose()

        beta, res, rank, sing = np.linalg.lstsq(X,B)
        t2s = 1 / beta[1, :].transpose()
        s0  = np.exp(beta[0, :]).transpose()

        t2s[np.isinf(t2s)] = 500.
        s0[np.isnan(s0)] = 0.

        t2ss[:, :, :, ne-2] = np.squeeze(unmask(t2s,mask))
        s0vs[:, :, :, ne-2] = np.squeeze(unmask(s0,mask))

    #Limited T2* and S0 maps
    fl = np.zeros([nx,ny,nz,len(tes)-2+1])
    for ne in range(Ne-1):
        fl_ = np.squeeze(fl[:,:,:,ne])
        fl_[masksum==ne+2] = True
        fl[:,:,:,ne] = fl_
    fl = np.array(fl,dtype=bool)
    t2sa = np.squeeze(unmask(t2ss[fl],masksum>1))

    return t2sa


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
