# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Align the fieldmap reference map to the target EPI.

The fieldmap reference map may be a magnitude image (or an EPI dataset,
in the case of PEPOLAR estimation).

The target EPI is the distorted dataset (or a reference thereof).

"""
from pkg_resources import resource_filename as pkgrf
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def init_coeff2epi_wf(
    omp_nthreads,
    debug=False,
    name="fmap2field_wf",
):
    """
    Move the field coefficients on to the target (distorted) EPI space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.apply.registration import init_coeff2epi_wf
            wf = init_coeff2epi_wf(omp_nthreads=2)

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
    target_ref
        the target EPI reference image
    target_mask
        the reference image (skull-stripped)
    fmap_ref
        the reference (anatomical) image corresponding to ``fmap``
    fmap_mask
        a brain mask corresponding to ``fmap``
    fmap_coeff
        fieldmap coefficients

    Outputs
    -------
    fmap_coeff
        fieldmap coefficients

    """
    from niworkflows.interfaces.fixes import FixHeaderRegistration as Registration
    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The estimated *fieldmap* was then aligned with rigid-registration to the target
EPI (echo-planar imaging) reference run.
The field coefficients were mapped on to the reference EPI using the transform.
"""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "target_ref",
                "target_mask",
                "fmap_ref",
                "fmap_mask",
                "fmap_coeff",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(niu.IdentityInterface(fields=["fmap_coeff"]),
                         name="outputnode")

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    ants_settings = pkgrf(
        "sdcflows", f"data/fmap-any_registration{'_testing' * debug}.json"
    )

    coregister = pe.Node(
        Registration(
            from_file=ants_settings,
        ),
        name="coregister",
        n_procs=omp_nthreads,
    )

    # Map the coefficients into the EPI space
    map_coeff = pe.Node(niu.Function(function=_move_coeff), name="map_coeff")

    # fmt: off
    workflow.connect([
        (inputnode, coregister, [
            ("target_ref", "fixed_image"),
            ("fmap_ref", "moving_image"),
            ("target_mask", "fixed_image_masks"),
            ("fmap_mask", "moving_image_masks"),
        ]),
        (inputnode, map_coeff, [("fmap_coeff", "in_coeff")]),
        (coregister, map_coeff, [("forward_transforms", "transform")]),
        (map_coeff, outputnode, [("out", "fmap_coeff")]),
    ])
    # fmt: on

    return workflow


def _move_coeff(in_coeff, transform):
    """Read in a rigid transform from ANTs, and update the coefficients field affine."""
    raise NotImplementedError
