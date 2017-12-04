# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Creating a T2*-map with multi-echo BOLD data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: echo_sampling_mask
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


def unmask(data, mask):
    """
    Unmasks `data` using non-zero entries of `mask`

    **Inputs**

    data
        Masked array of shape (nx*ny*nz[, Ne[, nt]])
    mask
        Boolean array of shape (nx, ny, nz)

    **Outputs**

    ndarray
        Array of shape (nx, ny, nz[, Ne[, nt]])

    """

    M = (mask != 0).ravel()
    Nm = M.sum()

    nx, ny, nz = mask.shape

    if len(data.shape) > 1:
        nt = data.shape[1]
    else:
        nt = 1

    out = np.zeros((nx * ny * nz, nt), dtype=data.dtype)
    out[M, :] = np.reshape(data, (Nm, nt))

    return np.squeeze(np.reshape(out, (nx, ny, nz, nt)))


def echo_sampling_mask(echo_list):
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
        two_echo_mask
            boolean array of voxels that can be sampled
            with at least two echos

    """
    # First, load each echo and average over time
    echos = []
    for e in echo_list:
        echos.append(np.mean(nib.load(e).get_data(), axis = -1))

    # In the first echo, find the 33rd percentile and the voxel(s)
    # whose average activity is equal to that value
    perc33  = np.percentile(echos[0][echos[0].nonzero()],
                            33, interpolation="higher")
    med_vox = (echos[0] == perc33)

    # For each (averaged) echo, extract the max signal in the
    # identified voxels and divide it by 3-- save as a threshold
    thrs = np.hstack([np.max(echo[med_vox]) / 3 for echo in echos])

    # Let's stack the arrays to make this next bit easier
    emeans = np.stack(echos, axis=-1)

    # Now, we want to find all voxels (in each echo) that show
    # absolute signal greater than our echo-specific threshold
    mthr = np.ones(emeans.shape)
    for i in range(emeans.shape[-1]):
        mthr[:, :, :, i] *= thrs[i]
    voxels = np.abs(emeans) > mthr

    # We save those voxel indices out to an array
    last_emask = np.array(voxels, dtype=np.int).sum(-1)
    # Save a mask of voxels sampled by at least two echos
    two_emask = (last_emask != 0)

    return last_emask, two_emask


def t2s_map(echo_list, last_emask, two_emask, tes):
    """

    **Inputs**

        echo_list
            List of file names for all echos
        last_emask
            numpy array where voxel values correspond to which
            echo a voxel can last be sampled with
        tes

        two_emask
            boolean array of voxels that can be sampled
            with at least two echos

    **Outputs**

        t2s_map

    """
    ############## UNRELATED ##############
    echos = []
    tes = [13.6, 29.79, 46.59]
    for e in echo_list:
        echos.append(nib.load(e).get_data())
    echo_stack = np.stack(echos, axis=-2)
    ############ END UNRELATED ############

    # get some basic shape information
    nx, ny, nz, necho, nt = echo_stack.shape
    nvox = nx * ny * nz

    # create empty arrays to fill later
    t2ss = np.zeros([nx, ny, nz, necho - 1])
    s0vs = t2ss.copy()

    # consider only those voxels sampled by at least two echos
    two_edata = echo_stack[two_emask.nonzero()]
    two_echo_nvox = two_edata.shape[0]

    # for the second echo on, do log linear fit
    for echo in range(2, necho + 1):

        # ΔS/S = ΔS0/S0 − ΔR2 * TE, so take neg TEs
        neg_tes = [-1 * te for te in tes[:echo]]

        # Create coefficient matrix
        a = np.array([np.ones(echo), neg_tes])
        A = np.tile(a, (1, nt))
        A = np.sort(A)[:, ::-1].transpose()

        # Create log-scale dependent-var matrix
        B = np.reshape(np.abs(two_edata[:, :echo, :]) + 1,
                       (two_echo_nvox, echo * nt)).transpose()
        B = np.log(B)

        # find the least squares solution for the echo
        X, res, rank, sing = np.linalg.lstsq(A, B)

        # scale the echo-coefficients (ΔR2), intercept (s0)
        r2 = 1 / X[1, :].transpose()
        s0  = np.exp(X[0, :]).transpose()

        # fix any non-numerical values
        r2[np.isinf(r2)] = 500.
        s0[np.isnan(s0)] = 0.

        # reshape into arrays for mapping
        t2s[:, :, :, echo - 2] = unmask(r2, two_emask)
        s0vs[:, :, :, echo - 2] = unmask(s0, two_emask)

    # limited T2* and S0 maps
    fl = np.zeros([nx, ny, nz, necho - 1])
    for echo in range(necho - 1):
        fl_ = fl[:, :, :, echo]
        fl_[last_emask == echo + 2] = True
        fl[:, :, :, echo] = fl_

    fl   = np.array(fl, dtype=bool)
    t2s_map = np.squeeze(unmask(t2s[fl], last_emask > 1))
    t2s_map[np.logical_or(np.isnan(t2s_map), t2s_map < 0)] = 0

    return t2s_map


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

    # inputnode = pe.Node(niu.IdentityInterface(fields=['t2svol', 'anat']),
    #                     name='inputnode')
    #
    # outputnode = pe.Node(niu.IdentityInterface(fields=['coreg_params']),
    #                      name='outputnode')

    get_thr = pe.Node(fsl.ImageStats(op_string='-P 50'),
                      name='get_thr')

    def format_expr(val):
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

    fmt_expr = pe.Node(name='fmt_expr',
                       interface=Function(input_names=['val'],
                                          output_names=['expr_string'],
                                          function=format_expr))

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
