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

from nipype.interfaces import ants, fsl, utility as niu
from nipype.pipeline import engine as pe
from niflow.nipype1.workflows.dmri.fsl.utils import (
    demean_image, cleanup_edge_pipeline)
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.images import IntraModalMerge
from niworkflows.interfaces.masks import BETRPT

from ..interfaces.fmap import Phasediff2Fieldmap, PhaseMap2rads


def init_phdiff_wf(omp_nthreads, name='phdiff_wf'):
    r"""
    Distortion correction of EPI sequences using phase-difference maps.

    Estimates the fieldmap using a phase-difference image and one or more
    magnitude images corresponding to two or more :abbr:`GRE (Gradient Echo sequence)`
    acquisitions.
    The most delicate bit of this workflow is the phase-unwrapping process: phase maps
    are clipped in the range :math:`[0 \dotsb 2 \cdot \pi )`.
    To find the integer number of offsets that make a region continously smooth with
    its neighbour, FSL PRELUDE is run [Jenkinson1998]_.
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

    """
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A deformation field to correct for susceptibility distortions was estimated
based on a field map that was co-registered to the BOLD reference,
using a custom workflow of *fMRIPrep* derived from D. Greve's `epidewarp.fsl`
[script](http://www.nmr.mgh.harvard.edu/~greve/fbirn/b0/epidewarp.fsl) and
further improvements of HCP Pipelines [@hcppipelines].
"""

    inputnode = pe.Node(niu.IdentityInterface(fields=['magnitude', 'phasediff', 'metadata']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap', 'fmap_ref', 'fmap_mask']), name='outputnode')

    # Merge input magnitude images
    magmrg = pe.Node(IntraModalMerge(), name='magmrg')

    # de-gradient the fields ("bias/illumination artifact")
    n4 = pe.Node(ants.N4BiasFieldCorrection(dimension=3, copy_header=True),
                 name='n4', n_procs=omp_nthreads)
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True),
                  name='bet')

    # uses mask from bet; outputs a mask
    # dilate = pe.Node(fsl.maths.MathsCommand(
    #     nan2zeros=True, args='-kernel sphere 5 -dilM'), name='MskDilate')

    # phase diff -> radians
    phmap2rads = pe.Node(PhaseMap2rads(), name='phmap2rads',
                         run_without_submitting=True)

    # FSL PRELUDE will perform phase-unwrapping
    prelude = pe.Node(fsl.PRELUDE(), name='prelude')

    recenter = pe.Node(niu.Function(function=_recenter),
                       name='recenter', run_without_submitting=True)

    denoise = pe.Node(fsl.SpatialFilter(operation='median', kernel_shape='sphere',
                                        kernel_size=3), name='denoise')

    demean = pe.Node(niu.Function(function=demean_image), name='demean')

    cleanup_wf = cleanup_edge_pipeline(name="cleanup_wf")

    compfmap = pe.Node(Phasediff2Fieldmap(), name='compfmap')

    workflow.connect([
        (inputnode, compfmap, [('metadata', 'metadata')]),
        (inputnode, magmrg, [('magnitude', 'in_files')]),
        (magmrg, n4, [('out_avg', 'input_image')]),
        (n4, prelude, [('output_image', 'magnitude_file')]),
        (n4, bet, [('output_image', 'in_file')]),
        (bet, prelude, [('mask_file', 'mask_file')]),
        (inputnode, phmap2rads, [('phasediff', 'in_file')]),
        (phmap2rads, prelude, [('out_file', 'phase_file')]),
        (prelude, recenter, [('unwrapped_phase_file', 'in_file')]),
        (recenter, denoise, [('out', 'in_file')]),
        (denoise, demean, [('out_file', 'in_file')]),
        (demean, cleanup_wf, [('out', 'inputnode.in_file')]),
        (bet, cleanup_wf, [('mask_file', 'inputnode.in_mask')]),
        (cleanup_wf, compfmap, [('outputnode.out_file', 'in_file')]),
        (compfmap, outputnode, [('out_file', 'fmap')]),
        (bet, outputnode, [('mask_file', 'fmap_mask'),
                           ('out_file', 'fmap_ref')]),
    ])

    return workflow


def _recenter(in_file, offset=None):
    """Recenter the phase-map distribution to the -pi..pi range."""
    from os.path import basename
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    if offset is None:
        offset = np.pi

    nii = nb.load(in_file)
    data = nii.get_fdata(dtype='float32') - offset

    out_file = fname_presuffix(basename(in_file), suffix='_recentered')
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(out_file)
    return out_file
