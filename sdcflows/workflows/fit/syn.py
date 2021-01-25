# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Estimating the susceptibility distortions without fieldmaps.

.. testsetup::

    >>> tmpdir = getfixture('tmpdir')
    >>> tmp = tmpdir.chdir() # changing to a temporary directory
    >>> data = np.zeros((10, 10, 10, 1, 3))
    >>> data[..., 1] = 1
    >>> nb.Nifti1Image(data, None, None).to_filename(
    ...     tmpdir.join('field.nii.gz').strpath)

.. _sdc_fieldmapless :

Fieldmap-less approaches
~~~~~~~~~~~~~~~~~~~~~~~~
Many studies acquired (especially with legacy MRI protocols) do not have any
information to estimate susceptibility-derived distortions.
In the absence of data with the specific purpose of estimating the :math:`B_0`
inhomogeneity map, researchers resort to nonlinear registration to an
«*anatomically correct*» map of the same individual (normally acquired with
:abbr:`T1w (T1-weighted)`, or :abbr:`T2w (T2-weighted)` sequences).
One of the most prominent proposals of this approach is found in [Studholme2000]_.

*SDCFlows* includes an (experimental) procedure (see :py:func:`init_syn_sdc_wf` below),
based on nonlinear image registration with ANTs' symmetric normalization (SyN) technique.
This workflow takes a skull-stripped :abbr:`T1w (T1-weighted)` image and
a reference :abbr:`EPI (Echo-Planar Imaging)` image, and estimates a field of nonlinear
displacements that accounts for susceptibility-derived distortions.
To more accurately estimate the warping on typically distorted regions, this
implementation uses an average :math:`B_0` mapping described in [Treiber2016]_.
The implementation is a variation on those developed in [Huntenburg2014]_ and
[Wang2017]_.
Feedback will be enthusiastically received.

References
----------
.. [Studholme2000] Studholme et al. (2000) Accurate alignment of functional EPI data to
    anatomical MRI using a physics-based distortion model,
    IEEE Trans Med Imag 19(11):1115-1127, 2000, doi: `10.1109/42.896788
    <https://doi.org/10.1109/42.896788>`__.
.. [Treiber2016] Treiber, J. M. et al. (2016) Characterization and Correction
    of Geometric Distortions in 814 Diffusion Weighted Images,
    PLoS ONE 11(3): e0152472. doi:`10.1371/journal.pone.0152472
    <https://doi.org/10.1371/journal.pone.0152472>`_.
.. [Wang2017] Wang S, et al. (2017) Evaluation of Field Map and Nonlinear
    Registration Methods for Correction of Susceptibility Artifacts
    in Diffusion MRI. Front. Neuroinform. 11:17.
    doi:`10.3389/fninf.2017.00017
    <https://doi.org/10.3389/fninf.2017.00017>`_.
.. [Huntenburg2014] Huntenburg, J. M. (2014) `Evaluating Nonlinear
    Coregistration of BOLD EPI and T1w Images
    <http://pubman.mpdl.mpg.de/pubman/item/escidoc:2327525:5/component/escidoc:2327523/master_thesis_huntenburg_4686947.pdf>`__,
    Berlin: Master Thesis, Freie Universität.

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

DEFAULT_MEMORY_MIN_GB = 0.01


def init_syn_sdc_wf(
    *,
    atlas_threshold=3,
    debug=False,
    name="syn_sdc_wf",
    omp_nthreads=1,
):
    """
    Build the *fieldmap-less* susceptibility-distortion estimation workflow.

    SyN deformation is restricted to the phase-encoding (PE) direction.
    If no PE direction is specified, anterior-posterior PE is assumed.

    SyN deformation is also restricted to regions that are expected to have a
    >3mm (approximately 1 voxel) warp, based on the fieldmap atlas.


    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.syn import init_syn_sdc_wf
            wf = init_syn_sdc_wf(omp_nthreads=8)

    Parameters
    ----------
    atlas_threshold : :obj:`float`
        Exclude from the registration metric computation areas with average distortions
        below this threshold (in mm).
    debug : :obj:`bool`
        Whether a fast (less accurate) configuration of the workflow should be applied.
    name : :obj:`str`
        Name for this workflow
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.

    Inputs
    ------
    epi_ref : :obj:`tuple` (:obj:`str`, :obj:`dict`)
        A tuple, where the first element is the path of the distorted EPI
        reference map (e.g., an average of *b=0* volumes), and the second
        element is a dictionary of associated metadata.
    epi_mask : :obj:`str`
        A path to a brain mask corresponding to ``epi_ref``.
    anat_brain : :obj:`str`
        A preprocessed, skull-stripped anatomical (T1w or T2w) image.
    std2anat_xfm : :obj:`str`
        inverse registration transform of T1w image to MNI template
    anat2epi_xfm : :obj:`str`
        transform mapping coordinates from the EPI space to the anatomical
        space (i.e., the transform to resample anatomical info into EPI space.)

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of files in ``epi_ref``.
    fmap_coeff : :obj:`str` or :obj:`list` of :obj:`str`
        The path(s) of the B-Spline coefficients supporting the fieldmap.

    """
    from pkg_resources import resource_filename as pkgrf
    from packaging.version import parse as parseversion, Version
    from nipype.interfaces.image import Rescale
    from niworkflows.interfaces.fixes import (
        FixHeaderApplyTransforms as ApplyTransforms,
        FixHeaderRegistration as Registration,
    )
    from niworkflows.interfaces.nibabel import Binarize
    from ...utils.misc import front as _pop
    from ...interfaces.utils import Deoblique, Reoblique
    from ...interfaces.bspline import (
        BSplineApprox,
        DEFAULT_LF_ZOOMS_MM,
        DEFAULT_HF_ZOOMS_MM,
        DEFAULT_ZOOMS_MM,
    )
    from ..ancillary import init_brainextraction_wf

    ants_version = Registration().version
    if ants_version and parseversion(ants_version) < Version("2.2.0"):
        raise RuntimeError(
            f"Please upgrade ANTs to 2.2 or older ({ants_version} found)."
        )

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
A deformation field to correct for susceptibility distortions was estimated
based on *fMRIPrep*'s *fieldmap-less* approach.
The deformation field is that resulting from co-registering the EPI reference
to the same-subject T1w-reference with its intensity inverted [@fieldmapless1;
@fieldmapless2].
Registration is performed with `antsRegistration`
(ANTs {ants_version or "-- version unknown"}), and
the process regularized by constraining deformation to be nonzero only
along the phase-encoding direction, and modulated with an average fieldmap
template [@fieldmapless3].
"""
    inputnode = pe.Node(
        niu.IdentityInterface(
            ["epi_ref", "epi_mask", "anat_brain", "std2anat_xfm", "anat2epi_xfm"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(["fmap", "fmap_ref", "fmap_coeff", "fmap_mask"]),
        name="outputnode",
    )

    invert_t1w = pe.Node(Rescale(invert=True), name="invert_t1w", mem_gb=0.3)
    anat2epi = pe.Node(
        ApplyTransforms(interpolation="BSpline"), name="anat2epi", n_procs=omp_nthreads
    )

    # Mapping & preparing prior knowledge
    # Concatenate transform files:
    # 1) anat -> EPI; 2) MNI -> anat; 3) ATLAS -> MNI
    transform_list = pe.Node(
        niu.Merge(3), name="transform_list", mem_gb=DEFAULT_MEMORY_MIN_GB
    )
    transform_list.inputs.in3 = pkgrf(
        "sdcflows", "data/fmap_atlas_2_MNI152NLin2009cAsym_affine.mat"
    )
    prior2epi = pe.Node(
        ApplyTransforms(input_image=pkgrf("sdcflows", "data/fmap_atlas.nii.gz")),
        name="prior2epi",
        n_procs=omp_nthreads,
        mem_gb=0.3,
    )
    atlas_msk = pe.Node(Binarize(thresh_low=atlas_threshold), name="atlas_msk")

    deoblique = pe.Node(Deoblique(), name="deoblique")
    reoblique = pe.Node(Reoblique(), name="reoblique")

    # SyN Registration Core
    syn = pe.Node(
        Registration(from_file=pkgrf("sdcflows", "data/susceptibility_syn.json")),
        name="syn",
        n_procs=omp_nthreads,
    )

    unwarp_ref = pe.Node(
        ApplyTransforms(interpolation="BSpline"),
        name="unwarp_ref",
    )

    brainextraction_wf = init_brainextraction_wf()

    # Extract nonzero component
    extract_field = pe.Node(niu.Function(function=_extract_field), name="extract_field")

    # Regularize with B-Splines
    bs_filter = pe.Node(BSplineApprox(), n_procs=omp_nthreads, name="bs_filter")
    bs_filter.interface._always_run = debug
    bs_filter.inputs.bs_spacing = (
        [DEFAULT_LF_ZOOMS_MM, DEFAULT_HF_ZOOMS_MM] if not debug else [DEFAULT_ZOOMS_MM]
    )
    bs_filter.inputs.extrapolate = not debug

    # fmt: off
    workflow.connect([
        (inputnode, transform_list, [("anat2epi_xfm", "in1"),
                                     ("std2anat_xfm", "in2")]),
        (inputnode, invert_t1w, [("anat_brain", "in_file"),
                                 (("epi_ref", _pop), "ref_file")]),
        (inputnode, anat2epi, [(("epi_ref", _pop), "reference_image"),
                               ("anat2epi_xfm", "transforms")]),
        (inputnode, deoblique, [(("epi_ref", _pop), "in_epi"),
                                ("epi_mask", "mask_epi")]),
        (inputnode, reoblique, [(("epi_ref", _pop), "in_epi")]),
        (inputnode, syn, [(("epi_ref", _warp_dir), "restrict_deformation")]),
        (inputnode, unwarp_ref, [(("epi_ref", _pop), "reference_image"),
                                 (("epi_ref", _pop), "input_image")]),
        (inputnode, prior2epi, [(("epi_ref", _pop), "reference_image")]),
        (inputnode, extract_field, [("epi_ref", "epi_meta")]),
        (invert_t1w, anat2epi, [("out_file", "input_image")]),
        (transform_list, prior2epi, [("out", "transforms")]),
        (prior2epi, atlas_msk, [("output_image", "in_file")]),
        (anat2epi, deoblique, [("output_image", "in_anat")]),
        (atlas_msk, deoblique, [("out_mask", "mask_anat")]),
        (deoblique, syn, [("out_epi", "moving_image"),
                          ("out_anat", "fixed_image"),
                          ("mask_epi", "moving_image_masks"),
                          (("mask_anat", _fixed_masks_arg), "fixed_image_masks")]),
        (syn, extract_field, [("forward_transforms", "in_file")]),
        (syn, unwarp_ref, [("forward_transforms", "transforms")]),
        (unwarp_ref, reoblique, [("output_image", "in_plumb")]),
        (reoblique, brainextraction_wf, [("out_epi", "inputnode.in_file")]),
        (extract_field, reoblique, [("out", "in_field")]),
        (reoblique, bs_filter, [("out_field", "in_data")]),
        (brainextraction_wf, bs_filter, [("outputnode.out_mask", "in_mask")]),
        (reoblique, outputnode, [("out_epi", "fmap_ref")]),
        (brainextraction_wf, outputnode, [("outputnode.out_mask", "fmap_mask")]),
        (bs_filter, outputnode, [
            ("out_extrapolated" if not debug else "out_field", "fmap"),
            ("out_coeff", "fmap_coeff")]),
    ])
    # fmt: on

    return workflow


def _warp_dir(intuple):
    """
    Extract the ``restrict_deformation`` argument from metadata.

    Example
    -------
    >>> _warp_dir(("epi.nii.gz", {"PhaseEncodingDirection": "i-"}))
    [[1, 0, 0], [1, 0, 0]]

    >>> _warp_dir(("epi.nii.gz", {"PhaseEncodingDirection": "j-"}))
    [[0, 1, 0], [0, 1, 0]]

    """
    pe = intuple[1]["PhaseEncodingDirection"][0]
    return 2 * [[int(pe == ax) for ax in "ijk"]]


def _fixed_masks_arg(mask):
    """
    Prepare the ``fixed_image_masks`` argument of SyN.

    Example
    -------
    >>> _fixed_masks_arg("atlas_mask.nii.gz")
    ['NULL', 'atlas_mask.nii.gz']

    """
    return ["NULL", mask]


def _extract_field(in_file, epi_meta):
    """
    Extract the nonzero component of the deformation field estimated by ANTs.

    Examples
    --------
    >>> nii = nb.load(
    ...     _extract_field(
    ...         ["field.nii.gz"],
    ...         ("epi.nii.gz", {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.005}))
    ... )
    >>> nii.shape
    (10, 10, 10)

    >>> np.allclose(nii.get_fdata(), -200)
    True

    """
    from nipype.utils.filemanip import fname_presuffix
    import numpy as np
    import nibabel as nb
    from sdcflows.utils.epimanip import get_trt

    fieldnii = nb.load(in_file[0])
    trt = get_trt(epi_meta[1], in_file=epi_meta[0])
    data = (
        np.squeeze(fieldnii.get_fdata(dtype="float32"))[
            ..., "ijk".index(epi_meta[1]["PhaseEncodingDirection"][0])
        ]
        / trt
        * (-1.0 if epi_meta[1]["PhaseEncodingDirection"].endswith("-") else 1.0)
    )
    out_file = fname_presuffix(in_file[0], suffix="_fieldmap")
    nii = nb.Nifti1Image(data, fieldnii.affine, None)
    nii.header.set_xyzt_units(fieldnii.header.get_xyzt_units()[0])
    nii.to_filename(out_file)
    return out_file
