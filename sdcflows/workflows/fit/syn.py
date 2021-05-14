# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
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
from pkg_resources import resource_filename
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

DEFAULT_MEMORY_MIN_GB = 0.01
INPUT_FIELDS = (
    "epi_ref",
    "epi_mask",
    "anat_ref",
    "anat_mask",
    "sd_prior",
)


def init_syn_sdc_wf(
    *,
    atlas_threshold=3,
    sloppy=False,
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
    sloppy : :obj:`bool`
        Whether a fast (less accurate) configuration of the workflow should be applied.
    debug : :obj:`bool`
        Run in debug mode
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
    anat_ref : :obj:`str`
        A preprocessed, skull-stripped anatomical (T1w or T2w) image resampled in EPI space.
    anat_mask : :obj:`str`
        Path to the brain mask corresponding to ``anat_ref`` in EPI space.
    sd_prior : :obj:`str`
        A template map of areas with strong susceptibility distortions (SD) to regularize
        the cost function of SyN

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of files in ``epi_ref``.
    fmap_coeff : :obj:`str` or :obj:`list` of :obj:`str`
        The path(s) of the B-Spline coefficients supporting the fieldmap.
    method: :obj:`str`
        Short description of the estimation method that was run.

    """
    from pkg_resources import resource_filename as pkgrf
    from packaging.version import parse as parseversion, Version
    from niworkflows.interfaces.fixes import (
        FixHeaderApplyTransforms as ApplyTransforms,
        FixHeaderRegistration as Registration,
    )
    from niworkflows.interfaces.header import CopyXForm
    from niworkflows.interfaces.nibabel import Binarize, RegridToZooms
    from ...utils.misc import front as _pop
    from ...interfaces.utils import Reoblique
    from ...interfaces.bspline import (
        BSplineApprox,
        DEFAULT_LF_ZOOMS_MM,
        DEFAULT_HF_ZOOMS_MM,
        DEFAULT_ZOOMS_MM,
    )
    from ...interfaces.brainmask import BinaryDilation, Union

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
    inputnode = pe.Node(niu.IdentityInterface(INPUT_FIELDS), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(["fmap", "fmap_ref", "fmap_coeff", "fmap_mask", "method"]),
        name="outputnode",
    )
    outputnode.inputs.method = 'FLB ("fieldmap-less", SyN-based)'

    warp_dir = pe.Node(
        niu.Function(function=_warp_dir),
        run_without_submitting=True,
        name="warp_dir",
    )

    atlas_msk = pe.Node(Binarize(thresh_low=atlas_threshold), name="atlas_msk")
    anat_dilmsk = pe.Node(BinaryDilation(), name="anat_dilmsk")
    epi_dilmsk = pe.Node(BinaryDilation(), name="epi_dilmsk")
    amask2epi = pe.Node(
        ApplyTransforms(interpolation="MultiLabel", transforms="identity"),
        name="amask2epi",
    )
    prior2epi = pe.Node(
        ApplyTransforms(interpolation="MultiLabel", transforms="identity"),
        name="prior2epi",
    )
    prior_dilmsk = pe.Node(BinaryDilation(radius=4), name="prior_dilmsk")

    epi_umask = pe.Node(Union(), name="epi_umask")
    moving_masks = pe.Node(
        niu.Merge(3),
        name="moving_masks",
        run_without_submitting=True,
    )

    fixed_masks = pe.Node(
        niu.Merge(3),
        name="fixed_masks",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )

    deoblique = pe.Node(CopyXForm(fields=["epi_ref"]), name="deoblique")
    reoblique = pe.Node(Reoblique(), name="reoblique")

    # Set a manageable size for the epi reference
    find_zooms = pe.Node(niu.Function(function=_adjust_zooms), name="find_zooms")
    zooms_epi = pe.Node(RegridToZooms(), name="zooms_epi")

    # SyN Registration Core
    syn = pe.Node(
        Registration(
            from_file=pkgrf("sdcflows", f"data/sd_syn{'_sloppy' * sloppy}.json")
        ),
        name="syn",
        n_procs=omp_nthreads,
    )
    syn.inputs.output_warped_image = debug
    syn.inputs.output_inverse_warped_image = debug

    if debug:
        syn.inputs.args = "--write-interval-volumes 5"

    unwarp_ref = pe.Node(
        ApplyTransforms(interpolation="BSpline"),
        name="unwarp_ref",
    )

    # Extract nonzero component
    extract_field = pe.Node(niu.Function(function=_extract_field), name="extract_field")

    # Check zooms (avoid very expensive B-Splines fitting)
    zooms_field = pe.Node(
        ApplyTransforms(interpolation="BSpline", transforms="identity"),
        name="zooms_field",
    )
    zooms_bmask = pe.Node(
        ApplyTransforms(interpolation="MultiLabel", transforms="identity"),
        name="zooms_bmask",
    )

    # Regularize with B-Splines
    bs_filter = pe.Node(BSplineApprox(), n_procs=omp_nthreads, name="bs_filter")
    bs_filter.interface._always_run = debug
    bs_filter.inputs.bs_spacing = (
        [DEFAULT_LF_ZOOMS_MM, DEFAULT_HF_ZOOMS_MM] if not sloppy else [DEFAULT_ZOOMS_MM]
    )
    bs_filter.inputs.extrapolate = not debug

    # fmt: off
    workflow.connect([
        (inputnode, extract_field, [("epi_ref", "epi_meta")]),
        (inputnode, atlas_msk, [("sd_prior", "in_file")]),
        (inputnode, deoblique, [(("epi_ref", _pop), "epi_ref"),
                                ("epi_mask", "hdr_file")]),
        (inputnode, reoblique, [(("epi_ref", _pop), "in_epi")]),
        (inputnode, epi_dilmsk, [("epi_mask", "in_file")]),
        (inputnode, zooms_bmask, [("anat_mask", "input_image")]),
        (inputnode, fixed_masks, [("anat_mask", "in2")]),
        (inputnode, anat_dilmsk, [("anat_mask", "in_file")]),
        (inputnode, warp_dir, [("epi_ref", "intuple")]),
        (inputnode, syn, [("anat_ref", "moving_image")]),
        (epi_dilmsk, prior2epi, [("out_file", "reference_image")]),
        (atlas_msk, prior2epi, [("out_file", "input_image")]),
        (prior2epi, prior_dilmsk, [("output_image", "in_file")]),
        (anat_dilmsk, fixed_masks, [("out_file", "in1")]),
        (warp_dir, syn, [("out", "restrict_deformation")]),
        (inputnode, find_zooms, [("anat_ref", "in_anat"),
                                 (("epi_ref", _pop), "in_epi")]),
        (deoblique, zooms_epi, [("epi_ref", "in_file")]),
        (deoblique, unwarp_ref, [("epi_ref", "input_image")]),
        (find_zooms, zooms_epi, [("out", "zooms")]),
        (zooms_epi, unwarp_ref, [("out_file", "reference_image")]),
        (atlas_msk, fixed_masks, [("out_mask", "in3")]),
        (fixed_masks, syn, [("out", "moving_image_masks")]),
        (epi_dilmsk, epi_umask, [("out_file", "in1")]),
        (epi_dilmsk, amask2epi, [("out_file", "reference_image")]),
        (anat_dilmsk, amask2epi, [("out_file", "input_image")]),
        (amask2epi, epi_umask, [("output_image", "in2")]),
        (epi_umask, moving_masks, [("out_file", "in1")]),
        (prior_dilmsk, moving_masks, [("out_file", "in2")]),
        (prior2epi, moving_masks, [("output_image", "in3")]),
        (moving_masks, syn, [("out", "fixed_image_masks")]),
        (deoblique, syn, [("epi_ref", "fixed_image")]),
        (syn, extract_field, [("reverse_transforms", "in_file")]),
        (syn, unwarp_ref, [("reverse_transforms", "transforms")]),
        (unwarp_ref, zooms_bmask, [("output_image", "reference_image")]),
        (unwarp_ref, zooms_field, [("output_image", "reference_image")]),
        (extract_field, zooms_field, [("out", "input_image")]),
        (unwarp_ref, reoblique, [("output_image", "in_plumb")]),
        (zooms_field, reoblique, [("output_image", "in_field")]),
        (zooms_bmask, reoblique, [("output_image", "in_mask")]),
        (reoblique, bs_filter, [("out_field", "in_data"),
                                ("out_mask", "in_mask")]),
        (reoblique, outputnode, [("out_epi", "fmap_ref"),
                                 ("out_mask", "fmap_mask")]),
        (bs_filter, outputnode, [
            ("out_extrapolated" if not debug else "out_field", "fmap"),
            ("out_coeff", "fmap_coeff")]),
    ])
    # fmt: on

    return workflow


def init_syn_preprocessing_wf(
    *,
    debug=False,
    name="syn_preprocessing_wf",
    omp_nthreads=1,
    auto_bold_nss=False,
    t1w_inversion=False,
):
    """
    Prepare EPI references and co-registration to anatomical for SyN.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.syn import init_syn_sdc_wf
            wf = init_syn_sdc_wf(omp_nthreads=8)

    Parameters
    ----------
    debug : :obj:`bool`
        Whether a fast (less accurate) configuration of the workflow should be applied.
    name : :obj:`str`
        Name for this workflow
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.
    auto_bold_nss : :obj:`bool`
        Set up the reference workflow to automatically execute nonsteady states detection
        of BOLD images.
    t1w_inversion : :obj:`bool`
        Run T1w intensity inversion so that it looks more like a T2 contrast.

    Inputs
    ------
    in_epis : :obj:`list` of :obj:`str`
        Distorted EPI images that will be merged together to create the
        EPI reference file.
    t_masks : :obj:`list` of :obj:`bool`
        (optional) mask of timepoints for calculating an EPI reference.
        Not used if ``auto_bold_nss=True``.
    in_meta : :obj:`list` of :obj:`dict`
        Metadata dictionaries corresponding to the ``in_epis`` input.
    in_anat : :obj:`str`
        A preprocessed anatomical (T1w or T2w) image.
    mask_anat : :obj:`str`
        A brainmask corresponding to the anatomical (T1w or T2w) image.
    std2anat_xfm : :obj:`str`
        inverse registration transform of T1w image to MNI template.

    Outputs
    -------
    epi_ref : :obj:`tuple` (:obj:`str`, :obj:`dict`)
        A tuple, where the first element is the path of the distorted EPI
        reference map (e.g., an average of *b=0* volumes), and the second
        element is a dictionary of associated metadata.
    anat_ref : :obj:`str`
        Path to the anatomical, skull-stripped reference in EPI space.
    anat_mask : :obj:`str`
        Path to the brain mask corresponding to ``anat_ref`` in EPI space.
    sd_prior : :obj:`str`
        A template map of areas with strong susceptibility distortions (SD) to regularize
        the cost function of SyN.

    """
    from pkg_resources import resource_filename as pkgrf
    from niworkflows.interfaces.nibabel import (
        IntensityClip,
        ApplyMask,
        GenerateSamplingReference,
    )
    from niworkflows.interfaces.fixes import (
        FixHeaderApplyTransforms as ApplyTransforms,
        FixHeaderRegistration as Registration,
    )
    from niworkflows.workflows.epi.refmap import init_epi_reference_wf
    from ...interfaces.utils import Deoblique, DenoiseImage
    from ...interfaces.brainmask import BrainExtraction, BinaryDilation

    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "in_epis",
                "t_masks",
                "in_meta",
                "in_anat",
                "mask_anat",
                "std2anat_xfm",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=["epi_ref", "epi_mask", "anat_ref", "anat_mask", "sd_prior"]
        ),
        name="outputnode",
    )

    deob_epi = pe.Node(Deoblique(), name="deob_epi")

    # Mapping & preparing prior knowledge
    # Concatenate transform files:
    # 1) MNI -> anat; 2) ATLAS -> MNI
    transform_list = pe.Node(
        niu.Merge(3),
        name="transform_list",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )
    transform_list.inputs.in3 = pkgrf(
        "sdcflows", "data/fmap_atlas_2_MNI152NLin2009cAsym_affine.mat"
    )
    prior2epi = pe.Node(
        ApplyTransforms(
            invert_transform_flags=[True, False, False],
            input_image=pkgrf("sdcflows", "data/fmap_atlas.nii.gz"),
        ),
        name="prior2epi",
        n_procs=omp_nthreads,
        mem_gb=0.3,
    )

    anat2epi = pe.Node(
        ApplyTransforms(invert_transform_flags=[True]),
        name="anat2epi",
        n_procs=omp_nthreads,
        mem_gb=0.3,
    )
    mask2epi = pe.Node(
        ApplyTransforms(invert_transform_flags=[True], interpolation="MultiLabel"),
        name="mask2epi",
        n_procs=omp_nthreads,
        mem_gb=0.3,
    )
    mask_dtype = pe.Node(
        niu.Function(function=_set_dtype, input_names=["in_file", "dtype"]),
        name="mask_dtype",
    )
    mask_dtype.inputs.dtype = "uint8"

    epi_reference_wf = init_epi_reference_wf(
        omp_nthreads=omp_nthreads,
        auto_bold_nss=auto_bold_nss,
    )
    epi_brain = pe.Node(BrainExtraction(), name="epi_brain")
    merge_output = pe.Node(
        niu.Function(function=_merge_meta),
        name="merge_output",
        run_without_submitting=True,
    )

    mask_anat = pe.Node(ApplyMask(), name="mask_anat")
    clip_anat = pe.Node(
        IntensityClip(p_min=0.0, p_max=99.8, invert=t1w_inversion), name="clip_anat"
    )
    ref_anat = pe.Node(DenoiseImage(copy_header=True), name="ref_anat",
                       n_procs=omp_nthreads)

    epi2anat = pe.Node(
        Registration(from_file=resource_filename("sdcflows", "data/affine.json")),
        name="epi2anat",
        n_procs=omp_nthreads,
    )
    epi2anat.inputs.output_warped_image = debug
    epi2anat.inputs.output_inverse_warped_image = debug
    if debug:
        epi2anat.inputs.args = "--write-interval-volumes 5"

    clip_anat_final = pe.Node(
        IntensityClip(p_min=0.0, p_max=100), name="clip_anat_final"
    )

    anat_dilmsk = pe.Node(BinaryDilation(), name="anat_dilmsk")
    epi_dilmsk = pe.Node(BinaryDilation(), name="epi_dilmsk")

    sampling_ref = pe.Node(GenerateSamplingReference(), name="sampling_ref")

    # fmt:off
    workflow.connect([
        (inputnode, transform_list, [("std2anat_xfm", "in2")]),
        (inputnode, epi_reference_wf, [("in_epis", "inputnode.in_files")]),
        (inputnode, merge_output, [("in_meta", "meta_list")]),
        (inputnode, anat_dilmsk, [("mask_anat", "in_file")]),
        (inputnode, mask_anat, [("in_anat", "in_file"),
                                ("mask_anat", "in_mask")]),
        (inputnode, mask2epi, [("mask_anat", "input_image")]),
        (epi_reference_wf, merge_output, [("outputnode.epi_ref_file", "epi_ref")]),
        (epi_reference_wf, deob_epi, [("outputnode.epi_ref_file", "in_file")]),
        (mask_anat, clip_anat, [("out_file", "in_file")]),
        (deob_epi, epi_brain, [("out_file", "in_file")]),
        (epi_brain, epi_dilmsk, [("out_mask", "in_file")]),
        (ref_anat, epi2anat, [("output_image", "fixed_image")]),
        (anat_dilmsk, epi2anat, [("out_file", "fixed_image_masks")]),
        (deob_epi, epi2anat, [("out_file", "moving_image")]),
        (epi_dilmsk, epi2anat, [("out_file", "moving_image_masks")]),
        (deob_epi, sampling_ref, [("out_file", "fixed_image")]),
        (epi2anat, transform_list, [("forward_transforms", "in1")]),
        (transform_list, prior2epi, [("out", "transforms")]),
        (sampling_ref, prior2epi, [("out_file", "reference_image")]),
        (epi2anat, anat2epi, [("forward_transforms", "transforms")]),
        (sampling_ref, anat2epi, [("out_file", "reference_image")]),
        (ref_anat, anat2epi, [("output_image", "input_image")]),
        (epi2anat, mask2epi, [("forward_transforms", "transforms")]),
        (sampling_ref, mask2epi, [("out_file", "reference_image")]),
        (mask2epi, mask_dtype, [("output_image", "in_file")]),
        (anat2epi, clip_anat_final, [("output_image", "in_file")]),
        (merge_output, outputnode, [("out", "epi_ref")]),
        (epi_brain, outputnode, [("out_mask", "epi_mask")]),
        (clip_anat_final, outputnode, [("out_file", "anat_ref")]),
        (mask_dtype, outputnode, [("out", "anat_mask")]),
        (prior2epi, outputnode, [("output_image", "sd_prior")]),
    ])
    # fmt:on

    if t1w_inversion:
        # Mask out non-brain zeros.
        mask_inverted = pe.Node(ApplyMask(), name="mask_inverted")
        # fmt:off
        workflow.connect([
            (inputnode, mask_inverted, [("mask_anat", "in_mask")]),
            (clip_anat, mask_inverted, [("out_file", "in_file")]),
            (mask_inverted, ref_anat, [("out_file", "input_image")]),
        ])
        # fmt:on
    else:
        # fmt:off
        workflow.connect([
            (clip_anat, ref_anat, [("out_file", "input_image")]),
        ])
        # fmt:on

    if debug:
        from niworkflows.interfaces.nibabel import RegridToZooms

        regrid_anat = pe.Node(
            RegridToZooms(zooms=(2.0, 2.0, 2.0), smooth=True), name="regrid_anat"
        )
        # fmt:off
        workflow.connect([
            (inputnode, regrid_anat, [("in_anat", "in_file")]),
            (regrid_anat, sampling_ref, [("out_file", "moving_image")]),
        ])
        # fmt:on
    else:
        # fmt:off
        workflow.connect([
            (inputnode, sampling_ref, [("in_anat", "moving_image")]),
        ])
        # fmt:on

    if not auto_bold_nss:
        workflow.connect(inputnode, "t_masks", epi_reference_wf, "inputnode.t_masks")

    return workflow


def _warp_dir(intuple, nlevels=3):
    """
    Extract the ``restrict_deformation`` argument from metadata.

    Example
    -------
    >>> _warp_dir(("epi.nii.gz", {"PhaseEncodingDirection": "i-"}))
    [[1, 0, 0], [1, 0, 0], [1, 0, 0]]

    >>> _warp_dir(("epi.nii.gz", {"PhaseEncodingDirection": "j-"}), nlevels=2)
    [[0, 1, 0], [0, 1, 0]]

    """
    pe = intuple[1]["PhaseEncodingDirection"][0]
    return nlevels * [[int(pe == ax) for ax in "ijk"]]


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
    from pathlib import Path
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
    out_file = Path(fname_presuffix(Path(in_file[0]).name, suffix="_fieldmap"))
    nii = nb.Nifti1Image(data, fieldnii.affine, None)
    nii.header.set_xyzt_units(fieldnii.header.get_xyzt_units()[0])
    nii.to_filename(out_file)
    return str(out_file.absolute())


def _merge_meta(epi_ref, meta_list):
    """Prepare a tuple of EPI reference and metadata."""
    return (epi_ref, meta_list[0])


def _set_dtype(in_file, dtype="int16"):
    """Change the dtype of an image."""
    import numpy as np
    import nibabel as nb

    img = nb.load(in_file)
    if img.header.get_data_dtype() == np.dtype(dtype):
        return in_file

    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(in_file, suffix=f"_{dtype}")
    hdr = img.header.copy()
    hdr.set_data_dtype(dtype)
    img.__class__(img.dataobj, img.affine, hdr).to_filename(out_file)
    return out_file


def _adjust_zooms(in_anat, in_epi, z_max=2.2, z_min=1.8):
    import nibabel as nb

    anat_res = min(nb.load(in_anat).header.get_zooms()[:3])
    epi_res = min(nb.load(in_epi).header.get_zooms()[:3])
    zoom_iso = min(
        round(max(0.5 * (anat_res + epi_res), z_min), 2),
        z_max,
    )
    return tuple([zoom_iso] * 3)
