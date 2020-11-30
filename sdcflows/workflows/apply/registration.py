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
    omp_nthreads, debug=False, write_coeff=False, name="fmap2field_wf",
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
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use.
    debug : :obj:`bool`
        Run fast configurations of registrations.
    name : :obj:`str`
        Unique name of this workflow.
    write_coeff : :obj:`bool`
        Map coefficients file

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
        fieldmap coefficients in the space of the target reference EPI
    target_ref
        the target reference EPI resampled into the fieldmap reference for
        quality control purposes.

    """
    from packaging.version import parse as parseversion, Version
    from niworkflows.interfaces.fixes import FixHeaderRegistration as Registration

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The estimated *fieldmap* was then aligned with rigid-registration to the target
EPI (echo-planar imaging) reference run.
The field coefficients were mapped on to the reference EPI using the transform.
"""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["target_ref", "target_mask", "fmap_ref", "fmap_mask", "fmap_coeff"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["target_ref", "fmap_coeff"]), name="outputnode"
    )

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    ants_settings = pkgrf(
        "sdcflows", f"data/fmap-any_registration{'_testing' * debug}.json"
    )

    coregister = pe.Node(
        Registration(from_file=ants_settings, output_warped_image=True,),
        name="coregister",
        n_procs=omp_nthreads,
    )

    ver = coregister.interface.version or "2.2.0"
    mask_trait_s = "s" if parseversion(ver) >= Version("2.2.0") else ""

    # fmt: off
    workflow.connect([
        (inputnode, coregister, [
            ("target_ref", "moving_image"),
            ("fmap_ref", "fixed_image"),
            ("target_mask", f"moving_image_mask{mask_trait_s}"),
            ("fmap_mask", f"fixed_image_mask{mask_trait_s}"),
        ]),
        (coregister, outputnode, [("warped_image", "target_ref")]),
    ])
    # fmt: on

    if not write_coeff:
        return workflow

    # Map the coefficients into the EPI space
    map_coeff = pe.Node(niu.Function(function=_move_coeff), name="map_coeff")
    map_coeff.interface._always_run = debug

    # fmt: off
    workflow.connect([
        (inputnode, map_coeff, [("fmap_coeff", "in_coeff"),
                                ("fmap_ref", "fmap_ref"),
                                ("target_ref", "target_ref")]),
        (coregister, map_coeff, [("forward_transforms", "transform")]),
        (map_coeff, outputnode, [("out", "fmap_coeff")]),
    ])
    # fmt: on

    return workflow


def _move_coeff(in_coeff, target_ref, fmap_ref, transform):
    """Read in a rigid transform from ANTs, and update the coefficients field affine."""
    from pathlib import Path
    import nibabel as nb
    import nitransforms as nt

    if isinstance(in_coeff, str):
        in_coeff = [in_coeff]

    xfm = nt.linear.Affine(
        nt.io.itk.ITKLinearTransform.from_filename(transform[0]).to_ras(),
        reference=fmap_ref,
    )
    xfm.apply(target_ref).to_filename("transformed.nii.gz")

    out = []
    for i, c in enumerate(in_coeff):
        img = nb.load(c)

        out.append(str(Path(f"moved_coeff_{i:03d}.nii.gz").absolute()))

        newaff = xfm.matrix @ img.affine
        img.__class__(img.dataobj, newaff, img.header).to_filename(out[-1])

    return out
