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

from nipype.interfaces import fsl, afni, utility as niu
from nipype.pipeline import engine as pe
from niflow.nipype1.workflows.dmri.fsl.utils import siemens2rads
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ..interfaces.fmap import Phasediff2Fieldmap, ProcessPhases, DiffPhases
from .gre import init_prepare_magnitude_wf, init_fmap_postproc_wf


def init_calculate_phasediff_wf(omp_nthreads, name='create_phasediff_wf'):
    """
    Create a phasediff image from two phase images.

    When two phase images are provided, the images are rescaled to range from 0 to 2*pi,
    then unwrapped using FSL's PRELUDE. Finally, the short TE phase image is subtracted from
    the long TE phase image. This workflow is based on the FSL FUGUE user guide.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from sdcflows.workflows.phdiff import init_calculate_phasediff_wf
        wf = init_calculate_phasediff_wf(omp_nthreads=1)

    **Parameters**:

        omp_nthreads : int
            Maximum number of threads an individual process may use
        difference : str
            Either 'arctan' or 'unwrapped_subtraction'

    **Inputs**:

        phase1 : pathlike
            Path to one of the phase images.
        phase2 : pathlike
            Path to the another phase image.
        phase1_metadata : dict
            Metadata dictionary corresponding to the phase1 input
        phase2_metadata : dict
            Metadata dictionary corresponding to the phase2 input
        magnitude : pathlike
            Preprocessed magnitude image
        mask_file : pathlike
            Brain mask image

    **Outputs**:
        phasediff : pathlike
            A phasediff image created by subtracting two upwrapped phase images.
        phasediff_metadata : dict
            A dictionary containing the metadata for the calculated ``phasediff``.
            It contains ``Echotime1`` and ``Echotime2`` from the original phase images.

    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['phase1', 'phase2', 'phase1_metadata', 'phase2_metadata', 'magnitude',
                    'mask_file']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['phasediff', 'phasediff_metadata']), name='outputnode')

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Individual phase images were unwrapped using FSL's PRELUDE and subtracted to
create a phasediff image."""

    # Scale to radians and unwrap phase1, phase2
    process_phases = pe.Node(ProcessPhases(), name='process_phases')
    unwrap_phase1 = pe.Node(fsl.PRELUDE(), name='unwrap_phase1')
    unwrap_phase2 = pe.Node(fsl.PRELUDE(), name='unwrap_phase2')
    subtract_phases = pe.Node(DiffPhases(), name='subtract_phases')

    workflow.connect([
        (inputnode, unwrap_phase1, [('magnitude', 'magnitude_file')]),
        (inputnode, unwrap_phase2, [('magnitude', 'magnitude_file')]),
        (inputnode, unwrap_phase1, [('mask_file', 'mask_file')]),
        (inputnode, unwrap_phase2, [('mask_file', 'mask_file')]),
        (process_phases, unwrap_phase1, [('short_te_phase_image', 'phase_file')]),
        (process_phases, unwrap_phase2, [('long_te_phase_image', 'phase_file')]),
        (unwrap_phase1, subtract_phases, [('unwrapped_phase_file', 'short_te_phase_image')]),
        (unwrap_phase2, subtract_phases, [('unwrapped_phase_file', 'long_te_phase_image')]),
        (subtract_phases, outputnode, [('phasediff_file', 'phasediff')]),
        (inputnode, process_phases, [
            ('phase1', 'phase1_file'),
            ('phase2', 'phase2_file'),
            ('phase1_metadata', 'phase1_metadata'),
            ('phase2_metadata', 'phase2_metadata')]),
        (process_phases, outputnode, [('phasediff_metadata', 'phasediff_metadata')])
    ])

    return workflow


def init_phdiff_wf(omp_nthreads, fmap_bspline, create_phasediff=False, name='phdiff_wf'):
    """
    Distortion correction of EPI sequences using phase-difference maps.

    Estimates the fieldmap using a phase-difference image and one or more
    magnitude images corresponding to two or more :abbr:`GRE (Gradient Echo sequence)`
    acquisitions. The `original code was taken from nipype
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
    phase1 : pathlike
        Path to one of the phase images.
    phase2 : pathlike
        Path to the another phase image.
    phase1_metadata : dict
        Metadata dictionary corresponding to the phase1 input
    phase2_metadata : dict
        Metadata dictionary corresponding to the phase2 input

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

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['magnitude', 'phasediff', 'metadata', 'phase1', 'phase2', 'phase1_metadata',
                    'phase2_metadata']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fmap', 'fmap_ref', 'fmap_mask']), name='outputnode')

    prepare_magnitude_wf = init_prepare_magnitude_wf(omp_nthreads=omp_nthreads)

    preprocessed_phasediff = pe.Node(
        niu.IdentityInterface(fields=['phasediff', 'phasediff_metadata']),
        name='preprocessed_phasediff')

    if create_phasediff:
        calc_phdiff_wf = init_calculate_phasediff_wf(omp_nthreads=omp_nthreads)
        workflow.connect([
            (inputnode, prepare_magnitude_wf, [('phase1', 'inputnode.source_file')]),
            (prepare_magnitude_wf, calc_phdiff_wf, [
                ('outputnode.fmap_mask', 'inputnode.mask_file')]),
            (prepare_magnitude_wf, calc_phdiff_wf, [
                ('outputnode.fmap_ref', 'inputnode.magnitude')]),
            (inputnode, calc_phdiff_wf, [
                ('phase1', 'inputnode.phase1'),
                ('phase2', 'inputnode.phase2'),
                ('phase1_metadata', 'inputnode.phase1_metadata'),
                ('phase2_metadata', 'inputnode.phase2_metadata')]),
            (calc_phdiff_wf, preprocessed_phasediff, [
                ('outputnode.phasediff', 'phasediff'),
                ('outputnode.phasediff_metadata', 'phasediff_metadata')])
        ])
        kernel_size = 5

    else:

        # FSL PRELUDE will perform phase-unwrapping
        prelude = pe.Node(fsl.PRELUDE(), name='prelude')

        # phase diff -> radians
        pha2rads = pe.Node(niu.Function(function=siemens2rads), name='pha2rads')

        workflow.connect([
            (inputnode, prepare_magnitude_wf, [('phasediff', 'inputnode.source_file')]),
            (prepare_magnitude_wf, prelude, [('outputnode.fmap_ref', 'magnitude_file')]),
            (prepare_magnitude_wf, prelude, [('outputnode.fmap_mask', 'mask_file')]),
            (inputnode, pha2rads, [('phasediff', 'in_file')]),
            (pha2rads, prelude, [('out', 'phase_file')]),
            (prelude, preprocessed_phasediff, [('unwrapped_phase_file', 'phasediff')]),
            (inputnode, preprocessed_phasediff, [('metadata', 'phasediff_metadata')]),
        ])
        kernel_size = 3

    fmap_postproc_wf = init_fmap_postproc_wf(omp_nthreads=omp_nthreads,
                                             median_kernel_size=kernel_size)

    compfmap = pe.Node(Phasediff2Fieldmap(), name='compfmap')

    # The phdiff2fmap interface is equivalent to:
    # rad2rsec (using rads2radsec from niflow.nipype1.workflows.dmri.fsl.utils)
    # pre_fugue = pe.Node(fsl.FUGUE(save_fmap=True), name='ComputeFieldmapFUGUE')
    # rsec2hz (divide by 2pi)

    workflow.connect([
        (inputnode, prepare_magnitude_wf, [('magnitude', 'inputnode.magnitude')]),
        (preprocessed_phasediff, fmap_postproc_wf, [('phasediff', 'inputnode.fmap')]),
        (prepare_magnitude_wf, fmap_postproc_wf, [
            ('outputnode.fmap_mask', 'inputnode.mask_file')]),
        (fmap_postproc_wf, compfmap, [('outputnode.out_fmap', 'in_file')]),
        (preprocessed_phasediff, compfmap, [('phasediff_metadata', 'metadata')]),
        (compfmap, outputnode, [('out_file', 'fmap')]),
        (prepare_magnitude_wf, outputnode, [
            ('outputnode.fmap_mask', 'fmap_mask'),
            ('outputnode.fmap_ref', 'fmap_ref')]),
    ])

    return workflow
