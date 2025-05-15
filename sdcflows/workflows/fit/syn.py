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
"""
import json

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import data

DEFAULT_MEMORY_MIN_GB = 0.01
INPUT_FIELDS = (
    "epi_ref",
    "epi_mask",
    "anat_ref",
    "anat_mask",
    "sd_prior",
)
MAX_LAPLACIAN_WEIGHT = 0.5


def init_syn_sdc_wf(
    *,
    use_metadata_estimates=False,
    fallback_total_readout_time=None,
    sloppy=False,
    debug=False,
    name="syn_sdc_wf",
    omp_nthreads=1,
    laplacian_weight=None,
    **kwargs,
):
    """
    Build the *fieldmap-less* susceptibility-distortion estimation workflow.

    SyN deformation is restricted to the phase-encoding (PE) direction.
    If no PE direction is specified, anterior-posterior PE is assumed.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.syn import init_syn_sdc_wf
            wf = init_syn_sdc_wf(omp_nthreads=8)

    Parameters
    ----------
    sloppy : :obj:`bool`
        Whether a fast (less accurate) configuration of the workflow should be applied.
    debug : :obj:`bool`
        Run in debug mode
    name : :obj:`str`
        Name for this workflow
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.
    laplacian_weight : :obj:`tuple` (optional)
        Tuple of two weights of the Laplacian term in the SyN cost function (one weight per
        registration level).
    sd_prior : :obj:`bool`
        Enable using a prior map to regularize the SyN cost function.

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
    from packaging.version import parse as parseversion, Version
    from nipype.interfaces.ants import ImageMath
    from niworkflows.interfaces.fixes import (
        FixHeaderApplyTransforms as ApplyTransforms,
        FixHeaderRegistration as Registration,
    )
    from niworkflows.interfaces.nibabel import (
        IntensityClip,
        RegridToZooms,
    )
    from ...utils.misc import front as _pop, last as _pull
    from ...interfaces.epi import GetReadoutTime
    from ...interfaces.fmap import DisplacementsField2Fieldmap
    from ...interfaces.bspline import (
        ApplyCoeffsField,
        BSplineApprox,
        DEFAULT_HF_ZOOMS_MM,
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
based on *SDCFlows*' *fieldmap-less* approach.
The deformation field is that resulting from co-registering the EPI reference
to the same-subject's T1w-reference [@fieldmapless1; @fieldmapless2].
Registration is performed with `antsRegistration`
(ANTs {ants_version or "-- version unknown"}), and
the process regularized by constraining deformation to be nonzero only
along the phase-encoding direction.
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
        GetReadoutTime(
            use_estimate=use_metadata_estimates,
        ),
        name="readout_time",
        run_without_submitting=True,
    )
    if fallback_total_readout_time is not None:
        readout_time.inputs.fallback = fallback_total_readout_time

    warp_dir = pe.Node(
        niu.Function(function=_warp_dir),
        run_without_submitting=True,
        name="warp_dir",
    )
    warp_dir.inputs.nlevels = 2
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
        niu.Merge(2),
        name="moving_masks",
        run_without_submitting=True,
    )
    moving_masks.inputs.in1 = "NULL"

    fixed_masks = pe.Node(
        niu.Merge(2),
        name="fixed_masks",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )

    # Set a manageable size for the epi reference
    find_zooms = pe.Node(niu.Function(function=_adjust_zooms), name="find_zooms")
    zooms_epi = pe.Node(RegridToZooms(), name="zooms_epi")

    syn_config = data.load(f"sd_syn{'_sloppy' * sloppy}.json")

    vox_params = pe.Node(niu.Function(function=_mm2vox), name="vox_params")
    vox_params.inputs.registration_config = json.loads(syn_config.read_text())

    # SyN Registration Core
    syn = pe.Node(
        Registration(from_file=syn_config),
        name="syn",
        n_procs=omp_nthreads,
    )
    syn.inputs.output_warped_image = debug
    syn.inputs.output_inverse_warped_image = debug

    if laplacian_weight is not None:
        laplacian_weight = (
            max(min(laplacian_weight[0], MAX_LAPLACIAN_WEIGHT), 0.0),
            max(min(laplacian_weight[1], MAX_LAPLACIAN_WEIGHT), 0.0),
        )
        syn.inputs.metric_weight = [
            [1.0 - laplacian_weight[0], laplacian_weight[0]],
            [1.0 - laplacian_weight[1], laplacian_weight[1]],
        ]

    if debug:
        syn.inputs.args = "--write-interval-volumes 2"

    # Extract the corresponding fieldmap in Hz
    extract_field = pe.Node(DisplacementsField2Fieldmap(), name="extract_field")

    unwarp = pe.Node(ApplyCoeffsField(jacobian=False), name="unwarp")

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
    bs_filter = pe.Node(
        BSplineApprox(recenter=False, debug=debug, extrapolate=not debug),
        name="bs_filter",
    )
    bs_filter.interface._always_run = debug
    bs_filter.inputs.bs_spacing = [DEFAULT_HF_ZOOMS_MM]

    if sloppy:
        bs_filter.inputs.zooms_min = 4.0

    workflow.connect([
        (inputnode, readout_time, [(("epi_ref", _pop), "in_file"),
                                   (("epi_ref", _pull), "metadata")]),
        (inputnode, clip_epi, [(("epi_ref", _pop), "in_file")]),
        (inputnode, unwarp, [(("epi_ref", _pop), "in_data")]),
        (inputnode, amask2epi, [("epi_mask", "reference_image")]),
        (inputnode, zooms_bmask, [("anat_mask", "input_image")]),
        (inputnode, fixed_masks, [("anat_mask", "in1"),
                                  ("sd_prior", "in2")]),
        (inputnode, anat_dilmsk, [("anat_mask", "in_file")]),
        (inputnode, warp_dir, [("anat_ref", "fixed_image")]),
        (inputnode, vox_params, [("anat_ref", "fixed_image")]),
        (inputnode, anat_merge, [("anat_ref", "in1")]),
        (inputnode, lap_anat, [("anat_ref", "op1")]),
        (inputnode, find_zooms, [("anat_ref", "in_anat"),
                                 (("epi_ref", _pop), "in_epi")]),
        (inputnode, zooms_field, [(("epi_ref", _pop), "reference_image")]),
        (inputnode, epi_umask, [("epi_mask", "in1")]),
        (lap_anat, lap_anat_norm, [("output_image", "in_file")]),
        (lap_anat_norm, anat_merge, [("out", "in2")]),
        (epi_umask, moving_masks, [("out_file", "in2")]),
        (clip_epi, epi_merge, [("out_file", "in1")]),
        (clip_epi, lap_epi, [("out_file", "op1")]),
        (clip_epi, zooms_epi, [("out_file", "in_file")]),
        (lap_epi, lap_epi_norm, [("output_image", "in_file")]),
        (lap_epi_norm, epi_merge, [("out", "in2")]),
        (find_zooms, zooms_epi, [("out", "zooms")]),
        (anat_dilmsk, amask2epi, [("out_file", "input_image")]),
        (amask2epi, epi_umask, [("output_image", "in2")]),
        (readout_time, warp_dir, [("pe_direction", "pe_dir")]),
        (readout_time, vox_params, [("pe_direction", "pe_dir")]),
        (clip_epi, warp_dir, [("out_file", "moving_image")]),
        (clip_epi, vox_params, [("out_file", "moving_image")]),
        (warp_dir, syn, [("out", "restrict_deformation")]),
        (anat_merge, syn, [("out", "fixed_image")]),
        (fixed_masks, syn, [("out", "fixed_image_masks")]),
        (epi_merge, syn, [("out", "moving_image")]),
        (moving_masks, syn, [("out", "moving_image_masks")]),
        (vox_params, syn, [("out", "transform_parameters")]),
        (syn, extract_field, [(("forward_transforms", _pop), "transform")]),
        (clip_epi, extract_field, [("out_file", "epi")]),
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
                              ("out_field", "fmap")]),
    ])  # fmt:skip

    return workflow


def init_syn_preprocessing_wf(
    *,
    atlas_threshold=3,
    debug=False,
    name="syn_preprocessing_wf",
    omp_nthreads=1,
    auto_bold_nss=False,
    t1w_inversion=False,
    sd_prior=True,
):
    """
    Prepare EPI references and co-registration to anatomical for SyN.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.syn import init_syn_preprocessing_wf
            wf = init_syn_preprocessing_wf()

    Parameters
    ----------
    atlas_threshold : :obj:`float`
        Mask excluding areas with average distortions below this threshold (in mm)
        on the prior.
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
    sd_prior : :obj:`bool`
        Enable using a prior map to regularize the SyN cost function.

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
        Registration(from_file=data.load("affine.json")),
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

    if sd_prior:
        from niworkflows.interfaces.nibabel import Binarize

        # Mapping & preparing prior knowledge
        # Concatenate transform files:
        # 1) MNI -> anat; 2) ATLAS -> MNI
        transform_list = pe.Node(
            niu.Merge(3),
            name="transform_list",
            mem_gb=DEFAULT_MEMORY_MIN_GB,
            run_without_submitting=True,
        )
        transform_list.inputs.in3 = data.load(
            "fmap_atlas_2_MNI152NLin2009cAsym_affine.mat"
        )
        prior2epi = pe.Node(
            ApplyTransforms(
                invert_transform_flags=[True, False, False],
                input_image=str(data.load("fmap_atlas.nii.gz")),
            ),
            name="prior2epi",
            n_procs=omp_nthreads,
            mem_gb=0.3,
        )

        prior_msk = pe.Node(Binarize(thresh_low=atlas_threshold), name="prior_msk")

        workflow.connect([
            (inputnode, transform_list, [("std2anat_xfm", "in2")]),
            (epi2anat, transform_list, [("forward_transforms", "in1")]),
            (transform_list, prior2epi, [("out", "transforms")]),
            (sampling_ref, prior2epi, [("out_file", "reference_image")]),
            (prior2epi, prior_msk, [("output_image", "in_file")]),
            (prior_msk, outputnode, [("out_mask", "sd_prior")]),
        ])  # fmt:skip

    else:
        # no prior to be used -> set anatomical mask as prior
        workflow.connect(mask_dtype, "out", outputnode, "sd_prior")

    workflow.connect([
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
    ])  # fmt:skip

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


def _warp_dir(moving_image, fixed_image, pe_dir, nlevels=2):
    """Extract the ``restrict_deformation`` argument from metadata."""
    import numpy as np
    import nibabel as nb

    moving = nb.load(moving_image)
    fixed = nb.load(fixed_image)

    if np.any(nb.affines.obliquity(fixed.affine) > 0.05):
        from nipype import logging

        logging.getLogger("nipype.interface").warn(
            "Running fieldmap-less registration on an oblique dataset"
        )

    moving_axcodes = nb.aff2axcodes(moving.affine, ["RR", "AA", "SS"])
    moving_pe_axis = moving_axcodes["ijk".index(pe_dir[0])]

    fixed_axcodes = nb.aff2axcodes(fixed.affine, ["RR", "AA", "SS"])

    deformation = [0.1, 0.1, 0.1]
    deformation[fixed_axcodes.index(moving_pe_axis)] = 1.0

    return nlevels * [deformation]


def _mm2vox(moving_image, fixed_image, pe_dir, registration_config):
    import nibabel as nb

    params = registration_config['transform_parameters']

    moving = nb.load(moving_image)
    # Use duplicate axcodes to ignore sign
    moving_axcodes = nb.aff2axcodes(moving.affine, ["RR", "AA", "SS"])
    moving_pe_axis = moving_axcodes["ijk".index(pe_dir[0])]

    fixed = nb.load(fixed_image)
    fixed_axcodes = nb.aff2axcodes(fixed.affine, ["RR", "AA", "SS"])

    zooms = nb.affines.voxel_sizes(fixed.affine)
    pe_res = zooms[fixed_axcodes.index(moving_pe_axis)]

    return [
        [*level_params[:2], level_params[2] / pe_res]
        for level_params in params
    ]


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
