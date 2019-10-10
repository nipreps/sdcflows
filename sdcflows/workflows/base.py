# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_base :

Automatic selection of the appropriate SDC method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the dataset metadata indicate tha more than one field map acquisition is
``IntendedFor`` (see BIDS Specification section 8.9) the following priority will
be used:

  1. :ref:`sdc_pepolar` (or **blip-up/blip-down**)

  2. :ref:`sdc_direct_b0`

  3. :ref:`sdc_phasediff`

  4. :ref:`sdc_fieldmapless`


Table of behavior (fieldmap use-cases):

=============== =========== ============= ===============
Fieldmaps found ``use_syn`` ``force_syn``     Action
=============== =========== ============= ===============
True            *           True          Fieldmaps + SyN
True            *           False         Fieldmaps
False           *           True          SyN
False           True        False         SyN
False           False       False         HMC only
=============== =========== ============= ===============


"""
from collections import defaultdict

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import logging

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.utility import KeySelect

# Fieldmap workflows
from .pepolar import init_pepolar_unwarp_wf
from .syn import init_syn_sdc_wf
from .unwarp import init_sdc_unwarp_wf

LOGGER = logging.getLogger('nipype.workflow')
FMAP_PRIORITY = {
    'epi': 0,
    'fieldmap': 1,
    'phasediff': 2,
    'syn': 3
}
DEFAULT_MEMORY_MIN_GB = 0.01


def init_sdc_wf(boldref, omp_nthreads=1, debug=False, ignore=None):
    """
    This workflow implements the heuristics to choose a
    :abbr:`SDC (susceptibility distortion correction)` strategy.
    When no field map information is present within the BIDS inputs,
    the EXPERIMENTAL "fieldmap-less SyN" can be performed, using
    the ``--use-syn`` argument. When ``--force-syn`` is specified,
    then the "fieldmap-less SyN" is always executed and reported
    despite of other fieldmaps available with higher priority.
    In the latter case (some sort of fieldmap(s) is available and
    ``--force-syn`` is requested), then the :abbr:`SDC (susceptibility
    distortion correction)` method applied is that with the
    highest priority.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from sdcflows.workflows.base import init_sdc_wf
        wf = init_sdc_wf(
            fmaps=[{
                'suffix': 'phasediff',
                'phasediff': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_phasediff.nii.gz',
                'magnitude1': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude1.nii.gz',
                'magnitude2': 'sub-03/ses-2/fmap/sub-03_ses-2_run-1_magnitude2.nii.gz',
            }],
            bold_meta={
                'RepetitionTime': 2.0,
                'SliceTiming': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'PhaseEncodingDirection': 'j',
            },
        )

    **Parameters**

        boldref : pybids.BIDSFile
            A BIDSFile object with suffix ``bold``, ``sbref`` or ``dwi``.
        omp_nthreads : int
            Maximum number of threads an individual process may use
        debug : bool
            Enable debugging outputs

    **Inputs**
        bold_ref
            A BOLD reference calculated at a previous stage
        bold_ref_brain
            Same as above, but brain-masked
        bold_mask
            Brain mask for the BOLD run
        t1_brain
            T1w image, brain-masked, for the fieldmap-less SyN method
        std2anat_xfm
            List of standard-to-T1w transforms generated during spatial
            normalization (only for the fieldmap-less SyN method).
        template : str
            Name of template from which prior knowledge will be mapped
            into the subject's T1w reference
            (only for the fieldmap-less SyN method)
        templates : str
            Name of templates that index the ``std2anat_xfm`` input list
            (only for the fieldmap-less SyN method).


    **Outputs**
        bold_ref
            An unwarped BOLD reference
        bold_mask
            The corresponding new mask after unwarping
        bold_ref_brain
            Brain-extracted, unwarped BOLD reference
        out_warp
            The deformation field to unwarp the susceptibility distortions
        syn_bold_ref
            If ``--force-syn``, an unwarped BOLD reference with this
            method (for reporting purposes)

    """

    if ignore is None:
        ignore = tuple()

    if not isinstance(ignore, (list, tuple)):
        ignore = tuple(ignore)

    fmaps = defaultdict(list, [])
    for associated in boldref.get_associations(kind='InformedBy'):
        if associated.suffix == 'epi':
            fmaps[associated.suffix].append(associated)
        # elif associated.suffix in ('phase', 'phasediff', 'fieldmap'):
        #     fmaps['fieldmap'].append(associated)

    workflow = Workflow(name='sdc_wf' if boldref else 'sdc_bypass_wf')
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_ref', 'bold_ref_brain', 'bold_mask',
                't1_brain', 'std2anat_xfm', 'template', 'templates']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['bold_ref', 'bold_mask', 'bold_ref_brain',
                'out_warp', 'syn_bold_ref', 'method']),
        name='outputnode')

    # No fieldmaps - forward inputs to outputs
    if not fmaps or 'fieldmaps' in ignore:
        workflow.__postdesc__ = """\
Susceptibility distortion correction (SDC) has been skipped because the
dataset does not contain extra field map acquisitions correctly described
with metadata, and the experimental SDC-SyN method was not explicitly selected.
"""
        outputnode.inputs.method = 'None'
        workflow.connect([
            (inputnode, outputnode, [('bold_ref', 'bold_ref'),
                                     ('bold_mask', 'bold_mask'),
                                     ('bold_ref_brain', 'bold_ref_brain')]),
        ])
        return workflow

    workflow.__postdesc__ = """\
Based on the estimated susceptibility distortion, an
unwarped BOLD reference was calculated for a more accurate
co-registration with the anatomical reference.
"""

    # PEPOLAR path
    if 'epi' in fmaps:
        outputnode.inputs.method = 'PEB/PEPOLAR (phase-encoding based / PE-POLARity)'
        # Get EPI polarities and their metadata
        sdc_unwarp_wf = init_pepolar_unwarp_wf(
            bold_meta=boldref.get_metadata(),
            epi_fmaps=[(fmap, fmap.get_metadata()["PhaseEncodingDirection"])
                       for fmap in fmaps['epi']],
            omp_nthreads=omp_nthreads,
            name='pepolar_unwarp_wf')

        workflow.connect([
            (inputnode, sdc_unwarp_wf, [
                ('bold_ref', 'inputnode.in_reference'),
                ('bold_mask', 'inputnode.in_mask'),
                ('bold_ref_brain', 'inputnode.in_reference_brain')]),
        ])

    # FIELDMAP path
    # elif 'fieldmap' in fmaps:
    #     # Import specific workflows here, so we don't break everything with one
    #     # unused workflow.
    #     suffices = {f.suffix for f in fmaps['fieldmap']}
    #     if 'fieldmap' in suffices:
    #         from .fmap import init_fmap_wf
    #         outputnode.inputs.method = 'FMB (fieldmap-based)'
    #         fmap_estimator_wf = init_fmap_wf(
    #             omp_nthreads=omp_nthreads,
    #             fmap_bspline=False)
    #         # set inputs
    #         fmap_estimator_wf.inputs.inputnode.fieldmap = fmap['fieldmap']
    #         fmap_estimator_wf.inputs.inputnode.magnitude = fmap['magnitude']

    #     if fmap['suffix'] == 'phasediff':
    #         from .phdiff import init_phdiff_wf
    #         fmap_estimator_wf = init_phdiff_wf(omp_nthreads=omp_nthreads)
    #         # set inputs
    #         fmap_estimator_wf.inputs.inputnode.phasediff = fmap['phasediff']
    #         fmap_estimator_wf.inputs.inputnode.magnitude = [
    #             fmap_ for key, fmap_ in sorted(fmap.items())
    #             if key.startswith("magnitude")
    #         ]

    #     sdc_unwarp_wf = init_sdc_unwarp_wf(
    #         omp_nthreads=omp_nthreads,
    #         fmap_demean=fmap_demean,
    #         debug=debug,
    #         name='sdc_unwarp_wf')
    #     sdc_unwarp_wf.inputs.inputnode.metadata = bold_meta

    #     workflow.connect([
    #         (inputnode, sdc_unwarp_wf, [
    #             ('bold_ref', 'inputnode.in_reference'),
    #             ('bold_ref_brain', 'inputnode.in_reference_brain'),
    #             ('bold_mask', 'inputnode.in_mask')]),
    #         (fmap_estimator_wf, sdc_unwarp_wf, [
    #             ('outputnode.fmap', 'inputnode.fmap'),
    #             ('outputnode.fmap_ref', 'inputnode.fmap_ref'),
    #             ('outputnode.fmap_mask', 'inputnode.fmap_mask')]),
    #     ])

    # # FIELDMAP-less path
    # if any(fm['suffix'] == 'syn' for fm in fmaps):
    #     # Select template
    #     sdc_select_std = pe.Node(KeySelect(
    #         fields=['std2anat_xfm']),
    #         name='sdc_select_std', run_without_submitting=True)

    #     syn_sdc_wf = init_syn_sdc_wf(
    #         bold_pe=bold_meta.get('PhaseEncodingDirection', None),
    #         omp_nthreads=omp_nthreads)

    #     workflow.connect([
    #         (inputnode, sdc_select_std, [
    #             ('template', 'key'),
    #             ('templates', 'keys'),
    #             ('std2anat_xfm', 'std2anat_xfm')]),
    #         (sdc_select_std, syn_sdc_wf, [
    #             ('std2anat_xfm', 'inputnode.std2anat_xfm')]),
    #         (inputnode, syn_sdc_wf, [
    #             ('t1_brain', 'inputnode.t1_brain'),
    #             ('bold_ref', 'inputnode.bold_ref'),
    #             ('bold_ref_brain', 'inputnode.bold_ref_brain'),
    #             ('template', 'inputnode.template')]),
    #     ])

    #     # XXX Eliminate branch when forcing isn't an option
    #     if fmap['suffix'] == 'syn':  # No fieldmaps, but --use-syn
    #         outputnode.inputs.method = 'FLB ("fieldmap-less", SyN-based)'
    #         sdc_unwarp_wf = syn_sdc_wf
    #     else:  # --force-syn was called when other fieldmap was present
    #         sdc_unwarp_wf.__desc__ = None
    #         workflow.connect([
    #             (syn_sdc_wf, outputnode, [
    #                 ('outputnode.out_reference', 'syn_bold_ref')]),
    #         ])

    workflow.connect([
        (sdc_unwarp_wf, outputnode, [
            ('outputnode.out_warp', 'out_warp'),
            ('outputnode.out_reference', 'bold_ref'),
            ('outputnode.out_reference_brain', 'bold_ref_brain'),
            ('outputnode.out_mask', 'bold_mask')]),
    ])

    return workflow
