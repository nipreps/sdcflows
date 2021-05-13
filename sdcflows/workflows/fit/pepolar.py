# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Datasets with multiple phase encoded directions.

.. _sdc_pepolar :

:abbr:`PEPOLAR (Phase Encoding POLARity)` techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This corresponds to `this section of the BIDS specification
<https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#case-4-multiple-phase-encoded-directions-pepolar>`__.

"""
from pkg_resources import resource_filename as _pkg_fname
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu

from niworkflows.engine.workflows import LiterateWorkflow as Workflow

INPUT_FIELDS = ("metadata", "in_data")
_PEPOLAR_DESC = """\
A *B<sub>0</sub>*-nonuniformity map (or *fieldmap*) was estimated based on two (or more)
echo-planar imaging (EPI) references """


def init_topup_wf(
    omp_nthreads=1, sloppy=False, debug=False, name="pepolar_estimate_wf"
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
    from niworkflows.interfaces.nibabel import MergeSeries
    from niworkflows.interfaces.images import IntraModalMerge

    from ...utils.misc import front as _front
    from ...interfaces.epi import GetReadoutTime
    from ...interfaces.utils import Flatten
    from ...interfaces.bspline import TOPUPCoeffReorient
    from ..ancillary import init_brainextraction_wf

    workflow = Workflow(name=name)
    workflow.__postdesc__ = f"""\
{_PEPOLAR_DESC} with `topup` (@topup; FSL {TOPUP().version}).
"""

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name="inputnode")
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
    outputnode.inputs.method = "PEB/PEPOLAR (phase-encoding based / PE-POLARity)"

    flatten = pe.Node(Flatten(), name="flatten")
    concat_blips = pe.Node(MergeSeries(), name="concat_blips")
    readout_time = pe.MapNode(
        GetReadoutTime(),
        name="readout_time",
        iterfield=["metadata", "in_file"],
        run_without_submitting=True,
    )

    topup = pe.Node(
        TOPUP(
            config=_pkg_fname("sdcflows", f"data/flirtsch/b02b0{'_quick' * sloppy}.cnf")
        ),
        name="topup",
    )
    merge_corrected = pe.Node(
        IntraModalMerge(hmc=False, to_ras=False), name="merge_corrected"
    )

    fix_coeff = pe.Node(
        TOPUPCoeffReorient(), name="fix_coeff", run_without_submitting=True
    )

    brainextraction_wf = init_brainextraction_wf()

    # fmt: off
    workflow.connect([
        (inputnode, flatten, [("in_data", "in_data"),
                              ("metadata", "in_meta")]),
        (flatten, readout_time, [("out_data", "in_file"),
                                 ("out_meta", "metadata")]),
        (flatten, concat_blips, [("out_data", "in_files")]),
        (readout_time, topup, [("readout_time", "readout_times"),
                               ("pe_dir_fsl", "encoding_direction")]),
        (concat_blips, topup, [("out_file", "in_file")]),
        (flatten, fix_coeff, [(("out_data", _front), "fmap_ref")]),
        (readout_time, topup, [(("pe_direction", _front), "pe_dir")]),
        (topup, fix_coeff, [("out_fieldcoef", "in_coeff")]),
        (topup, outputnode, [("out_jacs", "jacobians"),
                             ("out_mats", "xfms")]),
        (merge_corrected, brainextraction_wf, [("out_avg", "inputnode.in_file")]),
        (merge_corrected, outputnode, [("out_avg", "fmap_ref")]),
        (brainextraction_wf, outputnode, [("outputnode.out_mask", "fmap_mask")]),
        (fix_coeff, outputnode, [("out_coeff", "fmap_coeff")]),
    ])
    # fmt: on

    if not debug:
        # fmt: off
        workflow.connect([
            (topup, merge_corrected, [("out_corrected", "in_files")]),
            (topup, outputnode, [("out_field", "fmap"),
                                 ("out_warps", "out_warps")]),
        ])
        # fmt: on
        return workflow

    from ...interfaces.bspline import ApplyCoeffsField

    unwarp = pe.Node(ApplyCoeffsField(), name="unwarp")
    unwarp.interface._always_run = True

    # fmt:off
    workflow.connect([
        (fix_coeff, unwarp, [("out_coeff", "in_coeff")]),
        (flatten, unwarp, [("out_data", "in_target")]),
        (readout_time, unwarp, [("readout_time", "ro_time"),
                                ("pe_direction", "pe_dir")]),
        (unwarp, outputnode, [("out_warp", "out_warps"),
                              ("out_field", "fmap")]),
        (unwarp, merge_corrected, [("out_corrected", "in_files")]),
    ])
    # fmt:on

    return workflow


def init_3dQwarp_wf(omp_nthreads=1, debug=False, name="pepolar_estimate_wf"):
    """
    Create the PEPOLAR field estimation workflow based on AFNI's ``3dQwarp``.

    This workflow takes in two EPI files that MUST have opposed
    :abbr:`PE (phase-encoding)` direction.
    Therefore, EPIs with orthogonal PE directions are not supported.

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

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of the first element of ``in_data``.

    """
    from nipype.interfaces import afni
    from niworkflows.interfaces.header import CopyHeader
    from niworkflows.interfaces.fixes import (
        FixHeaderRegistration as Registration,
        FixHeaderApplyTransforms as ApplyTransforms,
    )
    from niworkflows.interfaces.freesurfer import StructuralReference
    from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
    from ...utils.misc import front as _front, last as _last
    from ...interfaces.utils import Flatten, ConvertWarp

    workflow = Workflow(name=name)
    workflow.__postdesc__ = f"""{_PEPOLAR_DESC} \
with `3dQwarp` (@afni; AFNI {''.join(['%02d' % v for v in afni.Info().version() or []])}).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_data", "metadata"]), name="inputnode"
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fmap", "fmap_ref"]), name="outputnode"
    )

    flatten = pe.Node(Flatten(), name="flatten")
    sort_pe = pe.Node(
        niu.Function(function=_sorted_pe, output_names=["sorted", "qwarp_args"]),
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
            from_file=_pkg_fname("sdcflows", "data/translation_rigid.json"),
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

    unwarp_reference = pe.Node(
        ApplyTransforms(
            dimension=3,
            float=True,
            interpolation="LanczosWindowedSinc",
        ),
        name="unwarp_reference",
    )

    # fmt: off
    workflow.connect([
        (inputnode, flatten, [("in_data", "in_data"),
                              ("metadata", "in_meta")]),
        (flatten, sort_pe, [("out_list", "inlist")]),
        (sort_pe, qwarp, [("qwarp_args", "args")]),
        (sort_pe, merge_pes, [("sorted", "in_files")]),
        (merge_pes, pe0_wf, [(("out_file", _front), "inputnode.in_file")]),
        (merge_pes, pe1_wf, [(("out_file", _last), "inputnode.in_file")]),
        (pe0_wf, align_pes, [("outputnode.skull_stripped_file", "fixed_image")]),
        (pe1_wf, align_pes, [("outputnode.skull_stripped_file", "moving_image")]),
        (pe0_wf, qwarp, [("outputnode.skull_stripped_file", "in_file")]),
        (align_pes, qwarp, [("warped_image", "base_file")]),
        (inputnode, cphdr_warp, [(("in_data", _front), "hdr_file")]),
        (qwarp, cphdr_warp, [("source_warp", "in_file")]),
        (cphdr_warp, to_ants, [("out_file", "in_file")]),
        (to_ants, unwarp_reference, [("out_file", "transforms")]),
        (inputnode, unwarp_reference, [("in_reference", "reference_image"),
                                       ("in_reference", "input_image")]),
        (unwarp_reference, outputnode, [("output_image", "fmap_ref")]),
        (to_ants, outputnode, [("out_file", "fmap")]),
    ])
    # fmt: on
    return workflow


def init_prepare_blips_wf(*, omp_nthreads=1, name="prepare_blips_wf"):
    """
    Prepare fieldmaps for PEPOLAR correction.

    This workflow takes in two or four EPI files.

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.
    name : :obj:`str`
        Name for this workflow


    Inputs
    ------
    epi_files : :obj:`list` of :obj:`str`
        A list of two or four EPI files, the first of which will be taken as reference.
    metadata : obj:`list` of :obj:`dict`
        A list of dictionaries containing the metadata corresponding to each file
        in ``epi_files``.

    Outputs
    -------
    reg_blips : :obj:`str`
        A 4D file containing one volume per phase-encoding direction

    """
    import pkg_resources as pkgr
    from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
    from niworkflows.interfaces.fixes import FixHeaderRegistration as Registration
    from niworkflows.interfaces.freesurfer import StructuralReference
    from niworkflows.interfaces.nibabel import MergeSeries
    from ...interfaces.utils import Flatten

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["epi_files", "metadata"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["reg_blips", "readout_times"]), name="outputnode"
    )

    flatten = pe.MapNode(Flatten(), name="flatten", iterfield=["in_data", "in_meta"])
    gen_pe_refs = pe.MapNode(
        StructuralReference(
            auto_detect_sensitivity=True,
            initial_timepoint=1,
            fixed_timepoint=True,  # Align to first image
            intensity_scaling=True,  # 7-DOF (rigid + intensity)
            no_iteration=True,
            subsample_threshold=200,
            transform_outputs=True,
            out_file="template.nii.gz",
        ),
        iterfield=["in_files"],
        name="gen_pe_refs",
    )
    n4_refs = pe.MapNode(
        N4BiasFieldCorrection(
            dimension=3,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            shrink_factor=4,
        ),
        n_procs=omp_nthreads,
        name="n4_refs",
        iterfield=["input_image"],
    )

    get_reg_files = pe.Node(
        niu.Function(function=_separate_first, output_names=["ref_image", "blips"]),
        name="get_reg_files",
    )

    reg_settings = pkgr.resource_filename("sdcflows", "data/translation_rigid.json")
    reg_blips = pe.MapNode(
        Registration(from_file=reg_settings, output_warped_image=True),
        name="reg_blips",
        iterfield=["moving_image"],
        n_procs=omp_nthreads,
    )

    concat_blips = pe.Node(niu.Merge(2), name="concat_blips")
    merge_blips = pe.Node(MergeSeries(), name="merge_blips")

    workflow = Workflow(name=name)
    # fmt: off
    workflow.connect([
        (inputnode, flatten, [
            ("epi_files", "in_data"),
            ("metadata", "in_meta")]),
        (flatten, gen_pe_refs, [("out_data", "in_files")]),
        (gen_pe_refs, n4_refs, [("out_file", "input_image")]),
        (n4_refs, get_reg_files, [("output_image", "in_files")]),
        (get_reg_files, reg_blips, [("ref_image", "fixed_image"),
                                    ("blips", "moving_image")]),
        (get_reg_files, concat_blips, [("ref_image", "in1")]),
        (reg_blips, concat_blips, [("warped_image", "in2")]),
        (concat_blips, merge_blips, [("out", "in_files")]),
        (merge_blips, outputnode, [("out_file", 'reg_blips')]),
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
            raise ValueError("Cannot handle orthogonal PE encodings.")

    return (
        [out_ref, out_opp],
        {"i": "-noYdis -noZdis", "j": "-noXdis -noZdis", "k": "-noXdis -noYdis"}[
            ref_pe[0]
        ],
    )


def _separate_first(in_files):
    """Take in a list of files and separate the first from the rest."""
    # TODO: check for best resolution image?
    if isinstance(in_files, (list, tuple)):
        return in_files[0], in_files[1:]
    raise RuntimeError(f"Expected an iterable but given {in_files}")
