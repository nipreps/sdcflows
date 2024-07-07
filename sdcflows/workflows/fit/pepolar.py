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
"""Datasets with multiple phase encoded directions."""
from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import data

INPUT_FIELDS = ("metadata", "in_data")
_PEPOLAR_DESC = """\
A *B<sub>0</sub>*-nonuniformity map (or *fieldmap*) was estimated based on two (or more)
echo-planar imaging (EPI) references """
_PEPOLAR_METHOD = "PEB/PEPOLAR (phase-encoding based / PE-POLARity)"


def init_topup_wf(
    grid_reference=0,
    omp_nthreads=1,
    sloppy=False,
    debug=False,
    name="pepolar_estimate_wf",
):
    """
    Create the PEPOLAR field estimation workflow based on FSL's ``topup``.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.pepolar import init_topup_wf
            wf = init_topup_wf()

    Parameters
    ----------
    grid_reference : :obj:`int`
        Index of the volume (after flattening) that will be taken for gridding reference.
    sloppy : :obj:`bool`
        Whether a fast configuration of topup (less accurate) should be applied.
    debug : :obj:`bool`
        Run in debug mode
    name : :obj:`str`
        Name for this workflow
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.

    Inputs
    ------
    in_data : :obj:`list` of :obj:`str`
        A list of EPI files that will be fed into TOPUP.
    metadata : :obj:`list` of :obj:`dict`
        A list of dictionaries containing the metadata corresponding to each file
        in ``in_data``.

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of files in ``in_data``.
    fmap_mask : :obj:`str`
        The path of mask corresponding to the ``fmap_ref`` output.
    fmap_coeff : :obj:`str` or :obj:`list` of :obj:`str`
        The path(s) of the B-Spline coefficients supporting the fieldmap.
    method: :obj:`str`
        Short description of the estimation method that was run.

    """
    from nipype.interfaces.fsl.epi import TOPUP
    from niworkflows.interfaces.images import RobustAverage
    from niworkflows.interfaces.nibabel import MergeSeries, ReorientImage

    from ...interfaces.bspline import TOPUPCoeffReorient
    from ...interfaces.epi import GetReadoutTime, SortPEBlips
    from ...interfaces.utils import (
        PadSlices,
        ReorientImageAndMetadata,
        UniformGrid,
    )
    from ...utils.misc import front as _front
    from ..ancillary import init_brainextraction_wf

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
{_PEPOLAR_DESC} with `topup` (@topup; FSL {TOPUP().version}).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=INPUT_FIELDS), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "jacobians",
                "xfms",
                "out_warps",
                "method",
            ]
        ),
        name="outputnode",
    )
    outputnode.inputs.method = _PEPOLAR_METHOD

    # Calculate the total readout time of each run
    readout_time = pe.MapNode(
        GetReadoutTime(),
        name="readout_time",
        iterfield=["metadata", "in_file"],
        run_without_submitting=True,
    )
    # Average each run so that topup is not overwhelmed (see #279)
    runwise_avg = pe.MapNode(
        RobustAverage(num_threads=omp_nthreads),
        name="runwise_avg",
        iterfield="in_file",
    )
    # Regrid all to the reference (grid_reference=0 means first averaged run)
    regrid = pe.Node(UniformGrid(reference=grid_reference), name="regrid")
    # Sort PE blips to ensure reproducibility
    sort_pe_blips = pe.Node(
        SortPEBlips(), name="sort_pe_blips", run_without_submitting=True
    )
    # Merge into one 4D file
    concat_blips = pe.Node(
        MergeSeries(affine_tolerance=1e-4), name="concat_blips"
    )
    # Pad dimensions so that they meet TOPUP's expectations
    pad_blip_slices = pe.Node(PadSlices(), name="pad_blip_slices")
    # Run 3dVolReg between runs: uses RobustAverage for consistency and to generate
    # debugging artifacts (typically, one wants to look at the average across uncorrected runs)
    setwise_avg = pe.Node(
        RobustAverage(num_threads=omp_nthreads), name="setwise_avg"
    )
    # The core of the implementation
    # Feed the input images in LAS orientation, so FSL does not run funky reorientations
    to_las = pe.Node(
        ReorientImageAndMetadata(target_orientation="LAS"), name="to_las"
    )
    topup = pe.Node(
        TOPUP(config=str(data.load(f"flirtsch/b02b0{'_quick' * sloppy}.cnf"))),
        name="topup",
    )
    # "Generalize" topup coefficients and store them in a spatially-correct NIfTI file
    fix_coeff = pe.Node(
        TOPUPCoeffReorient(), name="fix_coeff", run_without_submitting=True
    )

    # Average the output
    ref_average = pe.Node(
        RobustAverage(num_threads=omp_nthreads), name="ref_average"
    )

    # Sophisticated brain extraction of fMRIPrep
    brainextraction_wf = init_brainextraction_wf()

    # fmt: off
    workflow.connect([
        (inputnode, runwise_avg, [("in_data", "in_file")]),
        (inputnode, readout_time, [("metadata", "metadata")]),
        (runwise_avg, regrid, [("out_file", "in_data")]),
        (regrid, readout_time, [("out_data", "in_file")]),
        (regrid, sort_pe_blips, [("out_data", "in_data")]),
        (readout_time, sort_pe_blips, [("readout_time", "readout_times"),
                                       ("pe_dir_fsl", "pe_dirs_fsl")]),
        (sort_pe_blips, topup, [("readout_times", "readout_times")]),
        (setwise_avg, fix_coeff, [("out_file", "fmap_ref")]),
        (sort_pe_blips, concat_blips, [("out_data", "in_files")]),
        (concat_blips, pad_blip_slices, [("out_file", "in_file")]),
        (pad_blip_slices, setwise_avg, [("out_file", "in_file")]),
        (setwise_avg, to_las, [("out_hmc_volumes", "in_file")]),
        (sort_pe_blips, to_las, [("pe_dirs_fsl", "pe_dir")]),
        (to_las, topup, [
            ("out_file", "in_file"),
            ("pe_dir", "encoding_direction"),
        ]),
        (topup, fix_coeff, [("out_fieldcoef", "in_coeff")]),
        (to_las, fix_coeff, [(("pe_dir", _front), "pe_dir")]),
        (topup, outputnode, [("out_jacs", "jacobians"),
                             ("out_mats", "xfms")]),
        (ref_average, brainextraction_wf, [("out_file", "inputnode.in_file")]),
        (brainextraction_wf, outputnode, [
            ("outputnode.out_file", "fmap_ref"),
            ("outputnode.out_mask", "fmap_mask")
        ]),
        (fix_coeff, outputnode, [("out_coeff", "fmap_coeff")]),
    ])
    # fmt: on

    if not debug:
        # Roll orientation back to original
        from_las = pe.Node(ReorientImage(), name="from_las")
        from_las_fmap = pe.Node(ReorientImage(), name="from_las_fmap")
        # fmt: off
        workflow.connect([
            (setwise_avg, from_las, [("out_file", "target_file")]),
            (setwise_avg, from_las_fmap, [("out_file", "target_file")]),
            (topup, from_las, [("out_corrected", "in_file")]),
            (from_las, ref_average, [("out_file", "in_file")]),
            (topup, from_las_fmap, [("out_field", "in_file")]),
            (topup, outputnode, [("out_warps", "out_warps")]),
            (from_las_fmap, outputnode, [("out_file", "fmap")]),
        ])
        # fmt: on
        return workflow

    from sdcflows.interfaces.bspline import ApplyCoeffsField

    # Separate the runs again, as our ApplyCoeffsField corrects them separately
    unwarp = pe.Node(ApplyCoeffsField(), name="unwarp")
    unwarp.interface._always_run = True
    concat_corrected = pe.Node(MergeSeries(), name="concat_corrected")

    # fmt:off
    workflow.connect([
        (fix_coeff, unwarp, [("out_coeff", "in_coeff")]),
        (setwise_avg, unwarp, [("out_hmc_volumes", "in_data")]),
        (sort_pe_blips, unwarp, [("readout_times", "ro_time"),
                                 ("pe_dirs", "pe_dir")]),
        (unwarp, outputnode, [("out_field", "fmap")]),
        (unwarp, concat_corrected, [("out_corrected", "in_files")]),
        (concat_corrected, ref_average, [("out_file", "in_file")]),
    ])
    # fmt:on

    return workflow


def init_3dQwarp_wf(
    omp_nthreads=1, debug=False, sloppy=False, name="pepolar_estimate_wf"
):
    """
    Create the PEPOLAR field estimation workflow based on AFNI's ``3dQwarp``.

    This workflow takes in two EPI files that MUST have opposed
    :abbr:`PE (phase-encoding)` direction.
    Therefore, EPIs with orthogonal PE directions are not supported.
    ``3dQwarp`` is used to generate a displacement field and correct
    the reference image. The workflow also returns an estimated fieldmap,
    which is the result of converting the displacement field to a fieldmap
    and then regularizing it with a bspline field. This means that the unwarped
    image is in general not what one would get by reconstructing the fieldmap
    from fmap_coeff and warping the in_data.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.pepolar import init_3dQwarp_wf
            wf = init_3dQwarp_wf()

    Parameters
    ----------
    debug : :obj:`bool`
        Whether a fast configuration of topup (less accurate) should be applied.
    name : :obj:`str`
        Name for this workflow
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.

    Inputs
    ------
    in_data : :obj:`list` of :obj:`str`
        A list of two EPI files, the first of which will be taken as reference.
        The reference PhaseEncodingDirection should match the value for
        in_reference.
    metadata : :obj:`list` of :obj:`dict`
        A list with length matching the length of in_data. Each element should be a
        dict with keys that are strings and values of any type. One key should be
        PhaseEncodingDirection and the values should be BIDS-valid codings.

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of files in ``in_data``.
    fmap_mask : :obj:`str`
        The path of mask corresponding to the ``fmap_ref`` output.
    fmap_coeff : :obj:`str` or :obj:`list` of :obj:`str`
        The path(s) of the B-Spline coefficients supporting the fieldmap.
    method: :obj:`str`
        Short description of the estimation method that was run.
    out_warps: :obj:`str`
        The displacement field from 3dQwarp, in ANTS format.

    """
    from nipype.interfaces import afni
    from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
    from niworkflows.interfaces.fixes import (
        FixHeaderRegistration as Registration,
    )
    from niworkflows.interfaces.freesurfer import StructuralReference
    from niworkflows.interfaces.header import CopyHeader

    from ...interfaces.bspline import (
        DEFAULT_HF_ZOOMS_MM,
        DEFAULT_ZOOMS_MM,
        ApplyCoeffsField,
        BSplineApprox,
    )
    from ...interfaces.epi import GetReadoutTime
    from ...interfaces.fmap import DisplacementsField2Fieldmap
    from ...interfaces.utils import ConvertWarp, Flatten
    from ...utils.misc import front, last

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""{_PEPOLAR_DESC} \
with `3dQwarp` (@afni; AFNI {''.join(['%02d' % v for v in afni.Info().version() or []])}).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_data", "metadata"]), name="inputnode"
    )

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "fmap",
                "fmap_ref",
                "fmap_mask",
                "fmap_coeff",
                "method",
                "out_warps",
            ]
        ),
        name="outputnode",
    )
    outputnode.inputs.method = _PEPOLAR_METHOD

    readout_time = pe.Node(
        GetReadoutTime(),
        name="readout_time",
        run_without_submitting=True,
    )

    flatten = pe.Node(Flatten(), name="flatten")
    sort_pe = pe.Node(
        niu.Function(
            function=_sorted_pe, output_names=["sorted", "qwarp_args"]
        ),
        name="sort_pe",
        run_without_submitting=True,
    )

    merge_pes = pe.MapNode(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1,
            fixed_timepoint=True,  # Align to first image
            intensity_scaling=True,
            # 7-DOF (rigid + intensity)
            no_iteration=True,
            subsample_threshold=200,
            out_file="template.nii.gz",
        ),
        name="merge_pes",
        iterfield=["in_files"],
    )

    pe0_wf = init_enhance_and_skullstrip_bold_wf(
        omp_nthreads=omp_nthreads, name="pe0_wf"
    )
    pe1_wf = init_enhance_and_skullstrip_bold_wf(
        omp_nthreads=omp_nthreads, name="pe1_wf"
    )

    align_pes = pe.Node(
        Registration(
            from_file=data.load("translation_rigid.json"),
            output_warped_image=True,
        ),
        name="align_pes",
        n_procs=omp_nthreads,
    )

    qwarp = pe.Node(
        afni.QwarpPlusMinus(
            blur=[-1, -1],
            environ={"OMP_NUM_THREADS": f"{min(omp_nthreads, 4)}"},
            minpatch=9,
            nopadWARP=True,
            noweight=True,
            pblur=[0.05, 0.05],
        ),
        name="qwarp",
        n_procs=min(omp_nthreads, 4),
    )

    to_ants = pe.Node(ConvertWarp(), name="to_ants", mem_gb=0.01)

    cphdr_warp = pe.Node(CopyHeader(), name="cphdr_warp", mem_gb=0.01)

    # Extract the corresponding fieldmap in Hz
    extract_field = pe.Node(
        DisplacementsField2Fieldmap(), name="extract_field"
    )

    # Regularize with B-Splines
    bs_filter = pe.Node(
        BSplineApprox(debug=debug, extrapolate=not debug),
        name="bs_filter",
    )
    bs_filter.interface._always_run = debug
    bs_filter.inputs.bs_spacing = (
        [DEFAULT_HF_ZOOMS_MM] if not sloppy else [DEFAULT_ZOOMS_MM]
    )
    if sloppy:
        bs_filter.inputs.zooms_min = 4.0

    unwarp = pe.Node(ApplyCoeffsField(), name="unwarp")

    # fmt: off
    workflow.connect([
        (inputnode, flatten, [("in_data", "in_data"),
                              ("metadata", "in_meta")]),
        (flatten, sort_pe, [("out_list", "inlist")]),
        (sort_pe, qwarp, [("qwarp_args", "args")]),
        (sort_pe, merge_pes, [("sorted", "in_files")]),
        (merge_pes, pe0_wf, [(("out_file", front), "inputnode.in_file")]),
        (merge_pes, pe1_wf, [(("out_file", last), "inputnode.in_file")]),
        (pe0_wf, align_pes, [("outputnode.skull_stripped_file", "fixed_image")]),
        (pe1_wf, align_pes, [("outputnode.skull_stripped_file", "moving_image")]),
        (pe0_wf, qwarp, [("outputnode.skull_stripped_file", "in_file")]),
        (align_pes, qwarp, [("warped_image", "base_file")]),
        (inputnode, cphdr_warp, [(("in_data", front), "hdr_file")]),
        (qwarp, cphdr_warp, [("source_warp", "in_file")]),
        (cphdr_warp, to_ants, [("out_file", "in_file")]),
        (pe0_wf, extract_field, [("outputnode.skull_stripped_file", "epi")]),
        (to_ants, extract_field, [("out_file", "transform")]),
        (inputnode, readout_time, [(("metadata", front), "metadata")]),
        (readout_time, extract_field, [("readout_time", "ro_time"),
                                       ("pe_direction", "pe_dir")]),
        (pe1_wf, unwarp, [("outputnode.skull_stripped_file", "in_data")]),
        (pe0_wf, bs_filter, [("outputnode.mask_file", "in_mask")]),
        (extract_field, bs_filter, [("out_file", "in_data")]),
        (bs_filter, unwarp, [("out_coeff", "in_coeff")]),
        (readout_time, unwarp, [("readout_time", "ro_time"),
                                ("pe_direction", "pe_dir")]),
        (bs_filter, outputnode, [("out_coeff", "fmap_coeff")]),
        (qwarp, outputnode, [("warped_source", "fmap_ref")]),
        (unwarp, outputnode, [("out_field", "fmap")]),
        (pe0_wf, outputnode, [("outputnode.mask_file", "fmap_mask")]),
        (to_ants, outputnode, [("out_file", "out_warps")])

    ])
    # fmt: on
    return workflow


def _sorted_pe(inlist):
    """
    Generate suitable inputs to ``3dQwarp``.

    Example
    -------
    >>> paths, args = _sorted_pe([
    ...     ("dir-AP_epi.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-AP_bold.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-PA_epi.nii.gz", {"PhaseEncodingDirection": "j"}),
    ...     ("dir-PA_bold.nii.gz", {"PhaseEncodingDirection": "j"}),
    ...     ("dir-AP_sbref.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-PA_sbref.nii.gz", {"PhaseEncodingDirection": "j"}),
    ... ])
    >>> paths[0]
    ['dir-AP_epi.nii.gz', 'dir-AP_bold.nii.gz', 'dir-AP_sbref.nii.gz']

    >>> paths[1]
    ['dir-PA_epi.nii.gz', 'dir-PA_bold.nii.gz', 'dir-PA_sbref.nii.gz']

    >>> args
    '-noXdis -noZdis'

    >>> paths, args = _sorted_pe([
    ...     ("dir-AP_epi.nii.gz", {"PhaseEncodingDirection": "j-"}),
    ...     ("dir-LR_epi.nii.gz", {"PhaseEncodingDirection": "i"}),
    ... ])  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:

    """
    out_ref = [inlist[0][0]]
    out_opp = []

    ref_pe = inlist[0][1]["PhaseEncodingDirection"]
    for d, m in inlist[1:]:
        pe = m["PhaseEncodingDirection"]
        if pe == ref_pe:
            out_ref.append(d)
        elif pe[0] == ref_pe[0]:
            out_opp.append(d)
        else:
            msg = "Cannot handle orthogonal PE encodings."
            raise ValueError(msg)

    return (
        [out_ref, out_opp],
        {
            "i": "-noYdis -noZdis",
            "j": "-noXdis -noZdis",
            "k": "-noXdis -noYdis",
        }[ref_pe[0]],
    )
