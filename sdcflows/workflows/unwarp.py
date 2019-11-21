# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
.. _sdc_unwarp :

Unwarping
~~~~~~~~~

.. topic :: Abbreviations

    fmap
        fieldmap
    VSM
        voxel-shift map -- a 3D nifti where displacements are in pixels (not mm)
    DFM
        displacements field map -- a nifti warp file compatible with ANTs (mm)

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.images import FilledImageLike
from niworkflows.interfaces.registration import ANTSApplyTransformsRPT
from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf


def init_sdc_unwarp_wf(omp_nthreads, debug, name='sdc_unwarp_wf'):
    """
    Apply the warping given by a displacements fieldmap.

    This workflow takes in a displacements field through which the
    input reference can be corrected for susceptibility-derived distortion.

    It also calculates a new mask for the input dataset, which takes into
    account the (corrected) distortions.
    The mask is restricted to the field of view of the fieldmap since outside
    of it corrections could not be performed.

    .. workflow ::
        :graph2use: orig
        :simple_form: yes

        from sdcflows.workflows.unwarp import init_sdc_unwarp_wf
        wf = init_sdc_unwarp_wf(omp_nthreads=8,
                                debug=False)

    Parameters
    ----------
    omp_nthreads : int
        Maximum number of threads an individual process may use.
    debug : bool
        Run fast configurations of registrations.
    name : str
        Unique name of this workflow.

    Inputs
    ------
    in_warp : os.pathlike
        The :abbr:`DFM (displacements field map)` that corrects for
        susceptibility-derived distortions estimated elsewhere.
    in_reference
        the reference image to be unwarped.
    in_reference_brain
        a skull-stripped version corresponding to the ``in_reference``.

    Outputs
    -------
    out_reference
        the ``in_reference`` after unwarping
    out_reference_brain
        the ``in_reference`` after unwarping and skullstripping
    out_warp
        the ``in_warp`` field is forwarded for compatibility
    out_mask
        mask of the unwarped input file

    """
    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_warp', 'in_reference', 'in_reference_brain']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_reference', 'out_reference_brain', 'out_warp', 'out_mask']),
        name='outputnode')

    unwarp_reference = pe.Node(ANTSApplyTransformsRPT(dimension=3,
                                                      generate_report=False,
                                                      float=True,
                                                      interpolation='LanczosWindowedSinc'),
                               name='unwarp_reference')

    fieldmap_fov_mask = pe.Node(FilledImageLike(dtype='uint8'), name='fieldmap_fov_mask')
    fmap_fov2ref_apply = pe.Node(ANTSApplyTransformsRPT(
        generate_report=False, dimension=3, interpolation='NearestNeighbor',
        float=True),
        name='fmap_fov2ref_apply')

    apply_fov_mask = pe.Node(fsl.ApplyMask(), name="apply_fov_mask")
    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads,
                                                                         pre_mask=True)
    workflow.connect([
        (inputnode, unwarp_reference, [
            ('in_warp', 'transforms'),
            ('in_reference', 'reference_image'),
            ('in_reference', 'input_image')]),
        (inputnode, fieldmap_fov_mask, [('in_warp', 'in_file')]),
        (fieldmap_fov_mask, fmap_fov2ref_apply, [('out_file', 'input_image')]),
        (inputnode, fmap_fov2ref_apply, [('in_reference', 'reference_image')]),
        (fmap_fov2ref_apply, apply_fov_mask, [('output_image', 'mask_file')]),
        (unwarp_reference, apply_fov_mask, [('output_image', 'in_file')]),
        (apply_fov_mask, enhance_and_skullstrip_bold_wf, [('out_file', 'inputnode.in_file')]),
        (apply_fov_mask, outputnode, [('out_file', 'out_reference')]),
        (enhance_and_skullstrip_bold_wf, outputnode, [
            ('outputnode.mask_file', 'out_mask'),
            ('outputnode.skull_stripped_file', 'out_reference_brain')]),
        (inputnode, outputnode, [('in_warp', 'out_warp')]),
    ])
    return workflow
