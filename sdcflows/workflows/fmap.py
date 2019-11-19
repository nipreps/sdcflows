# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fieldmap-based estimation of susceptibility distortions.

.. _sdc_direct_b0 :

Direct B0 mapping sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~
When the fieldmap is directly measured with a prescribed sequence (such as
:abbr:`SE (spiral echo)`), we only need to calculate the corresponding B-Spline
coefficients to adapt the fieldmap to the TOPUP tool.
This procedure is described with more detail `here
<https://cni.stanford.edu/wiki/GE_Processing#Fieldmaps>`__.

This corresponds to `this section of the BIDS specification
<https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#a-real-fieldmap-image>`__.

"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu, fsl
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.images import IntraModalMerge

from .gre import init_fmap_postproc_wf, init_magnitude_wf


def init_fmap_wf(omp_nthreads, fmap_bspline, name='fmap_wf'):
    """
    Estimate the fieldmap based on a field-mapping MRI acquisition.

    When we have a sequence that directly measures the fieldmap,
    we just need to mask it (using the corresponding magnitude image)
    to remove the noise in the surrounding air region, and ensure that
    units are Hz.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fmap import init_fmap_wf
            wf = init_fmap_wf(omp_nthreads=6, fmap_bspline=False)

    Parameters
    ----------
    omp_nthreads : int
        Maximum number of threads an individual process may use.
    fmap_bspline : bool
        Whether the fieldmap estimate will be smoothed using BSpline basis.
    name : str
        Unique name of this workflow.

    Inputs
    ------
    magnitude : str
        Path to the corresponding magnitude image for anatomical reference.
    fieldmap : str
        Path to the fieldmap acquisition (``*_fieldmap.nii[.gz]`` of BIDS).

    Outputs
    -------
    fmap : str
        Path to the estimated fieldmap.
    fmap_ref : str
        Path to a preprocessed magnitude image reference.
    fmap_mask : str
        Path to a binary brain mask corresponding to the ``fmap`` and ``fmap_ref``
        pair.

    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['magnitude', 'fieldmap']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['fmap', 'fmap_ref', 'fmap_mask']),
                         name='outputnode')

    magnitude_wf = init_magnitude_wf(omp_nthreads=omp_nthreads)
    workflow.connect([
        (inputnode, magnitude_wf, [('magnitude', 'inputnode.magnitude')]),
        (magnitude_wf, outputnode, [('outputnode.fmap_mask', 'fmap_mask'),
                                    ('outputnode.fmap_ref', 'fmap_ref')]),
    ])

    # Merge input fieldmap images
    fmapmrg = pe.Node(IntraModalMerge(zero_based_avg=False, hmc=False),
                      name='fmapmrg')
    applymsk = pe.Node(fsl.ApplyMask(), name='applymsk')
    fmap_postproc_wf = init_fmap_postproc_wf(omp_nthreads=omp_nthreads,
                                             fmap_bspline=fmap_bspline)

    workflow.connect([
        (inputnode, fmapmrg, [('fieldmap', 'in_files')]),
        (fmapmrg, applymsk, [('out_file', 'in_file')]),
        (magnitude_wf, applymsk, [('outputnode.fmap_mask', 'mask_file')]),
        (applymsk, fmap_postproc_wf, [('out_file', 'inputnode.fmap')]),
        (magnitude_wf, fmap_postproc_wf, [
            ('outputnode.fmap_mask', 'inputnode.fmap_mask'),
            ('outputnode.fmap_ref', 'inputnode.fmap_ref')]),
        (fmap_postproc_wf, outputnode, [('outputnode.out_fmap', 'fmap')]),
    ])
    return workflow
