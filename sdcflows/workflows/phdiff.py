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

from nipype.interfaces import ants, fsl, afni, utility as niu
from nipype.pipeline import engine as pe
from niflow.nipype1.workflows.dmri.fsl.utils import (
    siemens2rads, demean_image, cleanup_edge_pipeline)
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.images import IntraModalMerge
from niworkflows.interfaces.masks import BETRPT

from ..interfaces.fmap import Phasediff2Fieldmap, ProcessPhases


def init_calculate_phasediff_wf(omp_nthreads, difference='unwrapped_subtraction',
                                name='create_phasediff_wf'):
    """
    Create a phasediff image from two phase images.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from sdcflows.workflows.phdiff import init_create_phasediff_wf
        wf = init_calculate_phasediff_wf(omp_nthreads=1)

    **Parameters**:

        omp_nthreads : int
            Maximum number of threads an individual process may use
        difference : str
            either 'arctan' or 'unwrapped_subtraction'
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
    """
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['phase1', 'phase2', 'phase1_metadata', 'phase2_metadata', 'magnitude',
                    'mask_file']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['phasediff', 'phasediff_metadata']), name='outputnode')

    workflow = Workflow(name=name)

    # Rescale images and calculate phasediff with arctan
    process_phases = pe.Node(ProcessPhases(), name='process_phases')
    unwrap_phase1 = pe.Node(fsl.PRELUDE(), name='unwrap_phase1')
    unwrap_phase2 = pe.Node(fsl.PRELUDE(), name='unwrap_phase2')
    subtract_phases = pe.Node(
        afni.Calc(outputtype='NIFTI_GZ', expr='b-a'), name='subtract_phases')

    workflow.connect([
        (inputnode, unwrap_phase1, [('magnitude', 'magnitude_file')]),
        (inputnode, unwrap_phase2, [('magnitude', 'magnitude_file')]),
        (inputnode, unwrap_phase1, [('mask_file', 'mask_file')]),
        (inputnode, unwrap_phase2, [('mask_file', 'mask_file')]),
        (process_phases, unwrap_phase1, [('short_te_phase_image', 'phase_file')]),
        (process_phases, unwrap_phase2, [('long_te_phase_image', 'phase_file')]),
        (unwrap_phase1, subtract_phases, [('unwrapped_phase_file', 'in_file_a')]),
        (unwrap_phase2, subtract_phases, [('unwrapped_phase_file', 'in_file_b')]),
        (subtract_phases, outputnode, [('out_file', 'phasediff')]),
        (inputnode, process_phases, [
            ('phase1', 'phase1_file'),
            ('phase2', 'phase2_file'),
            ('phase1_metadata', 'phase1_metadata'),
            ('phase2_metadata', 'phase2_metadata')]),
        (process_phases, outputnode, [('phasediff_metadata', 'phasediff_metadata')])
    ])

    return workflow


def init_phdiff_wf(omp_nthreads, create_phasediff=False, name='phdiff_wf'):
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

    preprocessed_phasediff = pe.Node(
        niu.IdentityInterface(fields=['phasediff', 'phasediff_metadata']),
        name='preprocessed_phasediff')

    if create_phasediff:
        calc_phdiff_wf = init_calculate_phasediff_wf(omp_nthreads=omp_nthreads)
        workflow.connect([
            (bet, calc_phdiff_wf, [('mask_file', 'inputnode.mask_file')]),
            (n4, calc_phdiff_wf, [('output_image', 'inputnode.magnitude')]),
            (inputnode, calc_phdiff_wf, [
                ('phase1', 'inputnode.phase1'),
                ('phase2', 'inputnode.phase2'),
                ('phase1_metadata', 'inputnode.phase1_metadata'),
                ('phase2_metadata', 'inputnode.phase2_metadata')]),
            (calc_phdiff_wf, preprocessed_phasediff, [
                ('outputnode.phasediff', 'phasediff'),
                ('outputnode.phasediff_metadata', 'phasediff_metadata')])
        ])

    else:

        # FSL PRELUDE will perform phase-unwrapping
        prelude = pe.Node(fsl.PRELUDE(), name='prelude')

        # phase diff -> radians
        pha2rads = pe.Node(niu.Function(function=siemens2rads), name='pha2rads')

        workflow.connect([
            (n4, prelude, [('output_image', 'magnitude_file')]),
            (bet, prelude, [('mask_file', 'mask_file')]),
            (inputnode, pha2rads, [('phasediff', 'in_file')]),
            (pha2rads, prelude, [('out', 'phase_file')]),
            (prelude, preprocessed_phasediff, [('unwrapped_phase_file', 'phasediff')]),
            (inputnode, preprocessed_phasediff, [('metadata', 'phasediff_metadata')]),
        ])

    denoise = pe.Node(fsl.SpatialFilter(operation='median', kernel_shape='sphere',
                                        kernel_size=3), name='denoise')

    demean = pe.Node(niu.Function(function=demean_image), name='demean')

    cleanup_wf = cleanup_edge_pipeline(name="cleanup_wf")

    compfmap = pe.Node(Phasediff2Fieldmap(), name='compfmap')

    # The phdiff2fmap interface is equivalent to:
    # rad2rsec (using rads2radsec from niflow.nipype1.workflows.dmri.fsl.utils)
    # pre_fugue = pe.Node(fsl.FUGUE(save_fmap=True), name='ComputeFieldmapFUGUE')
    # rsec2hz (divide by 2pi)

    workflow.connect([
        (inputnode, magmrg, [('magnitude', 'in_files')]),
        (magmrg, n4, [('out_avg', 'input_image')]),
        (n4, bet, [('output_image', 'in_file')]),
        (preprocessed_phasediff, denoise, [('phasediff', 'in_file')]),
        (denoise, demean, [('out_file', 'in_file')]),
        (demean, cleanup_wf, [('out', 'inputnode.in_file')]),
        (bet, cleanup_wf, [('mask_file', 'inputnode.in_mask')]),
        (cleanup_wf, compfmap, [('outputnode.out_file', 'in_file')]),
        (preprocessed_phasediff, compfmap, [('phasediff_metadata', 'metadata')]),
        (compfmap, outputnode, [('out_file', 'fmap')]),
        (bet, outputnode, [('mask_file', 'fmap_mask'),
                           ('out_file', 'fmap_ref')]),
    ])

    return workflow
