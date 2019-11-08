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
from nipype.interfaces import utility as niu, fsl, ants
from niflow.nipype1.workflows.dmri.fsl.utils import demean_image, cleanup_edge_pipeline
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.bids import DerivativesDataSink
from niworkflows.interfaces.images import IntraModalMerge
from niworkflows.interfaces.masks import BETRPT

from ..interfaces.fmap import (
    FieldEnhance, FieldToRadS, FieldToHz
)


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

    # Merge input magnitude images
    magmrg = pe.Node(IntraModalMerge(), name='magmrg')
    # Merge input fieldmap images
    fmapmrg = pe.Node(IntraModalMerge(zero_based_avg=False, hmc=False),
                      name='fmapmrg')

    # de-gradient the fields ("bias/illumination artifact")
    n4_correct = pe.Node(ants.N4BiasFieldCorrection(dimension=3, copy_header=True),
                         name='n4_correct', n_procs=omp_nthreads)
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True),
                  name='bet')
    ds_report_fmap_mask = pe.Node(DerivativesDataSink(
        desc='brain', suffix='mask'), name='ds_report_fmap_mask',
        run_without_submitting=True)

    workflow.connect([
        (inputnode, magmrg, [('magnitude', 'in_files')]),
        (inputnode, fmapmrg, [('fieldmap', 'in_files')]),
        (magmrg, n4_correct, [('out_file', 'input_image')]),
        (n4_correct, bet, [('output_image', 'in_file')]),
        (bet, outputnode, [('mask_file', 'fmap_mask'),
                           ('out_file', 'fmap_ref')]),
        (inputnode, ds_report_fmap_mask, [('fieldmap', 'source_file')]),
        (bet, ds_report_fmap_mask, [('out_report', 'in_file')]),
    ])

    if fmap_bspline:
        # despike_threshold=1.0, mask_erode=1),
        fmapenh = pe.Node(FieldEnhance(unwrap=False, despike=False),
                          name='fmapenh', mem_gb=4, n_procs=omp_nthreads)

        workflow.connect([
            (bet, fmapenh, [('mask_file', 'in_mask'),
                            ('out_file', 'in_magnitude')]),
            (fmapmrg, fmapenh, [('out_file', 'in_file')]),
            (fmapenh, outputnode, [('out_file', 'fmap')]),
        ])

    else:
        torads = pe.Node(FieldToRadS(), name='torads')
        prelude = pe.Node(fsl.PRELUDE(), name='prelude')
        tohz = pe.Node(FieldToHz(), name='tohz')

        denoise = pe.Node(fsl.SpatialFilter(operation='median', kernel_shape='sphere',
                                            kernel_size=3), name='denoise')
        demean = pe.Node(niu.Function(function=demean_image), name='demean')
        cleanup_wf = cleanup_edge_pipeline(name='cleanup_wf')

        applymsk = pe.Node(fsl.ApplyMask(), name='applymsk')

        workflow.connect([
            (bet, prelude, [('mask_file', 'mask_file'),
                            ('out_file', 'magnitude_file')]),
            (fmapmrg, torads, [('out_file', 'in_file')]),
            (torads, tohz, [('fmap_range', 'range_hz')]),
            (torads, prelude, [('out_file', 'phase_file')]),
            (prelude, tohz, [('unwrapped_phase_file', 'in_file')]),
            (tohz, denoise, [('out_file', 'in_file')]),
            (denoise, demean, [('out_file', 'in_file')]),
            (demean, cleanup_wf, [('out', 'inputnode.in_file')]),
            (bet, cleanup_wf, [('mask_file', 'inputnode.in_mask')]),
            (cleanup_wf, applymsk, [('outputnode.out_file', 'in_file')]),
            (bet, applymsk, [('mask_file', 'mask_file')]),
            (applymsk, outputnode, [('out_file', 'fmap')]),
        ])

    return workflow
