# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Applying a fieldmap given its B-Spline coefficients in Hz."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def init_unwarp_wf(omp_nthreads=1, debug=False, name="unwarp_wf"):
    """
    Set up a workflow that unwarps the input :abbr:`EPI (echo-planar imaging)` dataset.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.apply.correction import init_unwarp_wf
            wf = init_unwarp_wf(omp_nthreads=2)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use.
    name : :obj:`str`
        Unique name of this workflow.
    debug : :obj:`bool`
        Whether to run in *sloppy* mode.

    Inputs
    ------
    distorted
        the target EPI image.
    metadata
        dictionary of metadata corresponding to the target EPI image
    fmap_coeff
        fieldmap coefficients in distorted EPI space.

    Outputs
    -------
    fieldmap
        the displacements field interpolated from the B-Spline coefficients
        and scaled by the appropriate parameters (readout time of the EPI
        target and voxel size along PE).
    corrected
        the target EPI reference image, after applying unwarping.
    corrected_mask
        a fast mask calculated from the corrected EPI reference.

    """
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from ...interfaces.epi import GetReadoutTime
    from ...interfaces.bspline import Coefficients2Warp
    from ..ancillary import init_brainextraction_wf

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["distorted", "metadata", "fmap_coeff"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["corrected", "fieldmap", "corrected_mask"]),
        name="outputnode",
    )

    rotime = pe.Node(GetReadoutTime(), name="rotime")
    rotime.interface._always_run = debug
    resample = pe.Node(Coefficients2Warp(low_mem=debug), name="resample")
    unwarp = pe.Node(
        ApplyTransforms(dimension=3, interpolation="BSpline"), name="unwarp"
    )
    brainextraction_wf = init_brainextraction_wf()

    # fmt:off
    workflow.connect([
        (inputnode, rotime, [("distorted", "in_file"),
                             ("metadata", "metadata")]),
        (inputnode, resample, [("distorted", "in_target"),
                               ("fmap_coeff", "in_coeff")]),
        (rotime, resample, [("readout_time", "ro_time"),
                            ("pe_direction", "pe_dir")]),
        (inputnode, unwarp, [("distorted", "reference_image"),
                             ("distorted", "input_image")]),
        (resample, unwarp, [("out_warp", "transforms")]),
        (resample, outputnode, [("out_field", "fieldmap")]),
        (unwarp, brainextraction_wf, [("output_image", "inputnode.in_file")]),
        (brainextraction_wf, outputnode, [
            ("outputnode.out_file", "corrected"),
            ("outputnode.out_mask", "corrected_mask"),
        ]),
    ])
    # fmt:on
    return workflow
