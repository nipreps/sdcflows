# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Phase-difference B0 estimation.

.. _sdc_phasediff :

Phase-difference B0 estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The field inhomogeneity inside the scanner (fieldmap) is proportional to the
phase drift between two subsequent :abbr:`GRE (gradient recall echo)`
sequence.

This corresponds to `this section of the BIDS specification
<https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#two-phase-images-and-two-magnitude-images>`__.


"""

from nipype.interfaces import fsl, utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ..interfaces.fmap import Phasediff2Fieldmap, PhaseMap2rads
from .gre import init_fmap_postproc_wf, init_magnitude_wf


def init_phdiff_wf(omp_nthreads, name='phdiff_wf'):
    r"""
    Distortion correction of EPI sequences using phase-difference maps.

    Estimates the fieldmap using a phase-difference image and one or more
    magnitude images corresponding to two or more :abbr:`GRE (Gradient Echo sequence)`
    acquisitions.
    The most delicate bit of this workflow is the phase-unwrapping process: phase maps
    are clipped in the range :math:`[0 \dotsb 2 \cdot \pi )`.
    To find the integer number of offsets that make a region continously smooth with
    its neighbour, FSL PRELUDE is run [Jenkinson2003]_.
    FSL PRELUDE takes wrapped maps in the range 0 to 6.28, `as per the user guide
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#Step_2_-_Getting_.28wrapped.29_phase_in_radians>`__.
    For the phase-difference maps, recentering back to :math:`[-\pi \dotsb \pi )` is necessary.
    After some massaging and the application of the effective echo spacing parameter,
    the phase-difference maps can be converted into a *B0 field map* in Hz units.
    The `original code was taken from nipype
    <https://github.com/nipy/nipype/blob/0.12.1/nipype/workflows/dmri/fsl/artifacts.py#L514>`_.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.phdiff import init_phdiff_wf
            wf = init_phdiff_wf(omp_nthreads=1)

    Parameters
    ----------
    omp_nthreads : int
        Maximum number of threads an individual process may use

    Inputs
    ------
    magnitude : pathlike
        Path to the corresponding magnitude path(s).
    phasediff : pathlike
        Path to the corresponding phase-difference file.
    metadata : dict
        Metadata dictionary corresponding to the phasediff input

    Outputs
    -------
    fmap_ref : pathlike
        The average magnitude image, skull-stripped
    fmap_mask : pathlike
        The brain mask applied to the fieldmap
    fmap : pathlike
        The estimated fieldmap in Hz

    References
    ----------
    .. [Jenkinson2003] Jenkinson, M. (2003) Fast, automated, N-dimensional phase-unwrapping
        algorithm. MRM 49(1):193-197. doi:`10.1002/mrm.10354 <10.1002/mrm.10354>`__.

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A deformation field to correct for susceptibility distortions was estimated
based on a field map that was co-registered to the EPI (echo-planar imaging) reference
run, using a custom workflow of *SDCFlows* derived from D. Greve's `epidewarp.fsl`
[script](http://www.nmr.mgh.harvard.edu/~greve/fbirn/b0/epidewarp.fsl) and
further improvements of HCP Pipelines [@hcppipelines].
"""

    inputnode = pe.Node(niu.IdentityInterface(fields=['magnitude', 'phasediff', 'metadata']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap', 'fmap_ref', 'fmap_mask']), name='outputnode')

    magnitude_wf = init_magnitude_wf(omp_nthreads=omp_nthreads)

    # phase diff -> radians
    phmap2rads = pe.Node(PhaseMap2rads(), name='phmap2rads',
                         run_without_submitting=True)
    # FSL PRELUDE will perform phase-unwrapping
    prelude = pe.Node(fsl.PRELUDE(), name='prelude')

    fmap_postproc_wf = init_fmap_postproc_wf(omp_nthreads=omp_nthreads,
                                             fmap_bspline=False)
    compfmap = pe.Node(Phasediff2Fieldmap(), name='compfmap')

    workflow.connect([
        (inputnode, compfmap, [('metadata', 'metadata')]),
        (inputnode, magnitude_wf, [('magnitude', 'inputnode.magnitude')]),
        (magnitude_wf, prelude, [('outputnode.fmap_ref', 'magnitude_file'),
                                 ('outputnode.fmap_mask', 'mask_file')]),
        (inputnode, phmap2rads, [('phasediff', 'in_file')]),
        (phmap2rads, prelude, [('out_file', 'phase_file')]),
        (prelude, fmap_postproc_wf, [('unwrapped_phase_file', 'inputnode.fmap')]),
        (magnitude_wf, fmap_postproc_wf, [
            ('outputnode.fmap_mask', 'inputnode.fmap_mask'),
            ('outputnode.fmap_ref', 'inputnode.fmap_ref')]),
        (fmap_postproc_wf, compfmap, [('outputnode.out_fmap', 'in_file')]),
        (compfmap, outputnode, [('out_file', 'fmap')]),
        (magnitude_wf, outputnode, [('outputnode.fmap_ref', 'fmap_ref'),
                                    ('outputnode.fmap_mask', 'fmap_mask')]),
    ])

    return workflow
