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

*SDCFlows* includes an (experimental) procedure, based on nonlinear image registration
with ANTs' symmetric normalization (SyN) technique.
This workflow takes a skull-stripped :abbr:`T1w (T1-weighted)` image and
a reference :abbr:`EPI (Echo-Planar Imaging)` image, and estimates a field of nonlinear
displacements that accounts for susceptibility-derived distortions.
To more accurately estimate the warping on typically distorted regions, this
implementation uses an average :math:`B_0` mapping described in [Treiber2016]_.
The implementation is a variation on those developed in [Huntenburg2014]_ and
[Wang2017]_.

The process is divided in two steps.
First, the two images to be aligned (anatomical and one or more EPI sources) are prepared for
registration, including a linear pre-alignment of both, calculation of a 3D EPI *reference* map,
intensity/histogram enhancement, and *deobliquing* (meaning, for images were the physical
coordinates axes and the data array axes are not aligned, the physical coordinates are
rotated to align with the data array).
Such a preprocessing is implemented in :py:func:`init_syn_preprocessing_wf`.
Second, the outputs of the preprocessing workflow are fed into :py:func:`init_syn_sdc_wf`,
which executes the nonlinear, SyN registration.
To aid the *Mattes* mutual information cost function, the registration scheme is set up
in *multi-channel* mode, and laplacian-filtered derivatives of both anatomical and EPI
reference are introduced as a second registration channel.
The optimization gradients of the registration process are weighted, so that deformations
effectively possible only along the :abbr:`PE (phase-encoding)` axis.
Given that ANTs' registration framework performs on physical coordinates, it is necessary
that input images are not *oblique*.
The anatomical image is used as *fixed image*, and therefore, the registration process
estimates the transformation function from *unwarped* (anatomically *correct*) coordinates
to *distorted* coordinates.
If fed into ``antsApplyTransforms``, the resulting transform will effectively *unwarp* a distorted
EPI given as input into its *unwarped* mapping.
The estimated transform is then converted into a :math:`B_0` fieldmap in Hz, which can be
stored within the derivatives folder.

.. danger :: Experimental feature

    This procedure is experimental, and the outcomes should be scrutinized one-by-one
    and used with caution.
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
    out_warp : :obj:`str`
        The path of the corresponding displacements field transform to unwarp
        susceptibility distortions.
    method: :obj:`str`
        Short description of the estimation method that was run.

    """
    from pkg_resources import resource_filename as pkgrf
    from packaging.version import parse as parseversion, Version
    from nipype.interfaces.ants import ImageMath
    from niworkflows.interfaces.fixes import (
        FixHeaderApplyTransforms as ApplyTransforms,
        FixHeaderRegistration as Registration,
    )
    from niworkflows.interfaces.nibabel import (
        Binarize,
        IntensityClip,
        RegridToZooms,
    )
    from ...utils.misc import front as _pop, last as _pull
    from ...interfaces.epi import GetReadoutTime
    from ...interfaces.fmap import DisplacementsField2Fieldmap
    from ...interfaces.bspline import (
        ApplyCoeffsField,
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
        niu.IdentityInterface(
            ["fmap", "fmap_ref", "fmap_coeff", "fmap_mask", "out_warp", "method"]
        ),
        name="outputnode",
    )
    outputnode.inputs.method = 'FLB ("fieldmap-less", SyN-based)'

    readout_time = pe.Node(
        GetReadoutTime(),
        name="readout_time",
        run_without_submitting=True,
    )

    warp_dir = pe.Node(
        niu.Function(function=_warp_dir),
        run_without_submitting=True,
        name="warp_dir",
    )
    warp_dir.inputs.nlevels = 2
    atlas_msk = pe.Node(Binarize(thresh_low=atlas_threshold), name="atlas_msk")
    anat_dilmsk = pe.Node(BinaryDilation(), name="anat_dilmsk")
    amask2epi = pe.Node(
        ApplyTransforms(interpolation="MultiLabel", transforms="identity"),
        name="amask2epi",
    )

    # Calculate laplacian maps
    lap_anat = pe.Node(
        ImageMath(operation="Laplacian", op2="1.5 1", copy_header=True), name="lap_anat"
    )
    lap_anat_norm = pe.Node(niu.Function(function=_norm_lap), name="lap_anat_norm")
    anat_merge = pe.Node(
        niu.Merge(2),
        name="anat_merge",
        run_without_submitting=True,
    )

    clip_epi = pe.Node(IntensityClip(p_min=35.0, p_max=99.9), name="clip_epi")
    lap_epi = pe.Node(
        ImageMath(operation="Laplacian", op2="1.5 1", copy_header=True), name="lap_epi"
    )
    lap_epi_norm = pe.Node(niu.Function(function=_norm_lap), name="lap_epi_norm")
    epi_merge = pe.Node(
        niu.Merge(2),
        name="epi_merge",
        run_without_submitting=True,
    )

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
        syn.inputs.args = "--write-interval-volumes 2"

    # Extract the corresponding fieldmap in Hz
    extract_field = pe.Node(
        DisplacementsField2Fieldmap(demean=True), name="extract_field"
    )

    unwarp = pe.Node(ApplyCoeffsField(), name="unwarp")

    # Check zooms (avoid very expensive B-Splines fitting)
    zooms_field = pe.Node(
        ApplyTransforms(
            interpolation="BSpline", transforms="identity", args="-u float"
        ),
        name="zooms_field",
    )
    zooms_bmask = pe.Node(
        ApplyTransforms(
            interpolation="MultiLabel", transforms="identity", args="-u uchar"
        ),
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
        (inputnode, readout_time, [(("epi_ref", _pop), "in_file"),
                                   (("epi_ref", _pull), "metadata")]),
        (inputnode, atlas_msk, [("sd_prior", "in_file")]),
        (inputnode, clip_epi, [(("epi_ref", _pop), "in_file")]),
        (inputnode, unwarp, [(("epi_ref", _pop), "in_data")]),
        (inputnode, amask2epi, [("epi_mask", "reference_image")]),
        (inputnode, zooms_bmask, [("anat_mask", "input_image")]),
        (inputnode, fixed_masks, [("anat_mask", "in1"),
                                  ("anat_mask", "in2")]),
        (inputnode, anat_dilmsk, [("anat_mask", "in_file")]),
        (inputnode, warp_dir, [("anat_ref", "fixed_image")]),
        (inputnode, anat_merge, [("anat_ref", "in1")]),
        (inputnode, lap_anat, [("anat_ref", "op1")]),
        (inputnode, find_zooms, [("anat_ref", "in_anat"),
                                 (("epi_ref", _pop), "in_epi")]),
        (inputnode, zooms_field, [(("epi_ref", _pop), "reference_image")]),
        (inputnode, epi_umask, [("epi_mask", "in1")]),
        (lap_anat, lap_anat_norm, [("output_image", "in_file")]),
        (lap_anat_norm, anat_merge, [("out", "in2")]),
        (epi_umask, moving_masks, [("out_file", "in1"),
                                   ("out_file", "in2"),
                                   ("out_file", "in3")]),
        (clip_epi, epi_merge, [("out_file", "in1")]),
        (clip_epi, lap_epi, [("out_file", "op1")]),
        (clip_epi, zooms_epi, [("out_file", "in_file")]),
        (lap_epi, lap_epi_norm, [("output_image", "in_file")]),
        (lap_epi_norm, epi_merge, [("out", "in2")]),
        (find_zooms, zooms_epi, [("out", "zooms")]),
        (atlas_msk, fixed_masks, [("out_mask", "in3")]),
        (anat_dilmsk, amask2epi, [("out_file", "input_image")]),
        (amask2epi, epi_umask, [("output_image", "in2")]),
        (readout_time, warp_dir, [("pe_direction", "pe_dir")]),
        (warp_dir, syn, [("out", "restrict_deformation")]),
        (anat_merge, syn, [("out", "fixed_image")]),
        (fixed_masks, syn, [("out", "fixed_image_masks")]),
        (epi_merge, syn, [("out", "moving_image")]),
        (moving_masks, syn, [("out", "moving_image_masks")]),
        (syn, extract_field, [(("forward_transforms", _pop), "transform")]),
        (readout_time, extract_field, [("readout_time", "ro_time"),
                                       ("pe_direction", "pe_dir")]),
        (extract_field, zooms_field, [("out_file", "input_image")]),
        (zooms_field, zooms_bmask, [("output_image", "reference_image")]),
        (zooms_field, bs_filter, [("output_image", "in_data")]),
        # Setting a mask ends up over-fitting the field
        # - it's better to have all those ~zero around.
        # (zooms_bmask, bs_filter, [("output_image", "in_mask")]),
        (bs_filter, unwarp, [("out_coeff", "in_coeff")]),
        (readout_time, unwarp, [("readout_time", "ro_time"),
                                ("pe_direction", "pe_dir")]),
        (zooms_bmask, outputnode, [("output_image", "fmap_mask")]),
        (bs_filter, outputnode, [("out_coeff", "fmap_coeff")]),
        (unwarp, outputnode, [("out_corrected", "fmap_ref"),
                              ("out_field", "fmap"),
                              ("out_warp", "out_warp")]),
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
    clip_anat = pe.Node(IntensityClip(p_min=0.0, p_max=99.8), name="clip_anat")
    ref_anat = pe.Node(
        DenoiseImage(copy_header=True), name="ref_anat", n_procs=omp_nthreads
    )

    epi2anat = pe.Node(
        Registration(from_file=resource_filename("sdcflows", "data/affine.json")),
        name="epi2anat",
        n_procs=omp_nthreads,
    )
    epi2anat.inputs.output_warped_image = debug
    epi2anat.inputs.output_inverse_warped_image = debug
    if debug:
        epi2anat.inputs.args = "--write-interval-volumes 5"

    def _remove_first_mask(in_file):
        if not isinstance(in_file, list):
            in_file = [in_file]

        in_file.insert(0, "NULL")
        return in_file

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
        (epi_reference_wf, deob_epi, [("outputnode.epi_ref_file", "in_file")]),
        (deob_epi, merge_output, [("out_file", "epi_ref")]),
        (mask_anat, clip_anat, [("out_file", "in_file")]),
        (clip_anat, ref_anat, [("out_file", "input_image")]),
        (deob_epi, epi_brain, [("out_file", "in_file")]),
        (epi_brain, epi_dilmsk, [("out_mask", "in_file")]),
        (ref_anat, epi2anat, [("output_image", "fixed_image")]),
        (anat_dilmsk, epi2anat, [("out_file", "fixed_image_masks")]),
        (deob_epi, epi2anat, [("out_file", "moving_image")]),
        (epi_dilmsk, epi2anat, [
            (("out_file", _remove_first_mask), "moving_image_masks")]),
        (deob_epi, sampling_ref, [("out_file", "fixed_image")]),
        (epi2anat, transform_list, [("forward_transforms", "in1")]),
        (transform_list, prior2epi, [("out", "transforms")]),
        (sampling_ref, prior2epi, [("out_file", "reference_image")]),
        (ref_anat, anat2epi, [("output_image", "input_image")]),
        (epi2anat, anat2epi, [("forward_transforms", "transforms")]),
        (sampling_ref, anat2epi, [("out_file", "reference_image")]),
        (epi2anat, mask2epi, [("forward_transforms", "transforms")]),
        (sampling_ref, mask2epi, [("out_file", "reference_image")]),
        (mask2epi, mask_dtype, [("output_image", "in_file")]),
        (anat2epi, outputnode, [("output_image", "anat_ref")]),
        (mask_dtype, outputnode, [("out", "anat_mask")]),
        (merge_output, outputnode, [("out", "epi_ref")]),
        (epi_brain, outputnode, [("out_mask", "epi_mask")]),
        (prior2epi, outputnode, [("output_image", "sd_prior")]),
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


def _warp_dir(fixed_image, pe_dir, nlevels=3):
    """Extract the ``restrict_deformation`` argument from metadata."""
    import numpy as np
    import nibabel as nb

    img = nb.load(fixed_image)

    if np.any(nb.affines.obliquity(img.affine) > 0.05):
        from nipype import logging

        logging.getLogger("nipype.interface").warn(
            "Running fieldmap-less registration on an oblique dataset"
        )

    vs = nb.affines.voxel_sizes(img.affine)
    order = np.around(np.abs(img.affine[:3, :3] / vs))
    retval = order @ [1 if pe_dir[0] == ax else 0.1 for ax in "ijk"]

    return nlevels * [retval.tolist()]


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


def match_histogram(reference, image, ref_mask=None, img_mask=None):
    """Match the histogram of the T2-like anatomical with the EPI."""
    import os
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    from skimage.exposure import match_histograms

    nii_img = nb.load(image)
    img_data = np.asanyarray(nii_img.dataobj)
    ref_data = np.asanyarray(nb.load(reference).dataobj)

    ref_mask = (
        np.ones_like(ref_data, dtype=bool)
        if ref_mask is None
        else np.asanyarray(nb.load(ref_mask).dataobj) > 0
    )

    img_mask = (
        np.ones_like(img_data, dtype=bool)
        if img_mask is None
        else np.asanyarray(nb.load(img_mask).dataobj) > 0
    )

    out_file = fname_presuffix(image, suffix="_matched", newpath=os.getcwd())
    img_data[img_mask] = match_histograms(
        img_data[img_mask],
        ref_data[ref_mask],
    )

    nii_img.__class__(
        img_data,
        nii_img.affine,
        nii_img.header,
    ).to_filename(out_file)
    return out_file


def _norm_lap(in_file):
    """Brought over from nirodents."""
    from pathlib import Path
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    img = nb.load(in_file)
    data = img.get_fdata()
    data -= np.median(data)
    l_max = np.percentile(data[data > 0], 99.8)
    l_min = np.percentile(data[data < 0], 0.2)
    data[data < 0] *= -1.0 / l_min
    data[data > 0] *= 1.0 / l_max
    data = np.clip(data, a_min=-1.0, a_max=1.0)

    out_file = fname_presuffix(
        Path(in_file).name, suffix="_norm", newpath=str(Path.cwd().absolute())
    )
    hdr = img.header.copy()
    hdr.set_data_dtype("float32")
    img.__class__(data.astype("float32"), img.affine, hdr).to_filename(out_file)
    return out_file
