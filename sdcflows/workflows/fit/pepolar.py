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
    from niworkflows.interfaces.nibabel import MergeSeries
    from niworkflows.interfaces.images import RobustAverage

    from ...utils.misc import front as _front
    from ...interfaces.epi import GetReadoutTime
    from ...interfaces.utils import Flatten, UniformGrid
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
    regrid = pe.Node(UniformGrid(reference=grid_reference), name="regrid")
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
    ref_average = pe.Node(RobustAverage(), name="ref_average")

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
        (flatten, regrid, [("out_data", "in_data")]),
        (regrid, concat_blips, [("out_data", "in_files")]),
        (readout_time, topup, [("readout_time", "readout_times"),
                               ("pe_dir_fsl", "encoding_direction")]),
        (regrid, fix_coeff, [("reference", "fmap_ref")]),
        (readout_time, fix_coeff, [(("pe_direction", _front), "pe_dir")]),
        (topup, fix_coeff, [("out_fieldcoef", "in_coeff")]),
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
        # fmt: off
        workflow.connect([
            (concat_blips, topup, [("out_file", "in_file")]),
            (topup, ref_average, [("out_corrected", "in_file")]),
            (topup, outputnode, [("out_field", "fmap"),
                                 ("out_warps", "out_warps")]),
        ])
        # fmt: on
        return workflow

    from nipype.interfaces.afni.preprocess import Volreg
    from niworkflows.interfaces.nibabel import SplitSeries
    from ...interfaces.bspline import ApplyCoeffsField

    realign = pe.Node(
        Volreg(args=f"-base {grid_reference}", outputtype="NIFTI_GZ"),
        name="realign_blips",
    )
    split_blips = pe.Node(SplitSeries(), name="split_blips")
    unwarp = pe.Node(ApplyCoeffsField(), name="unwarp")
    unwarp.interface._always_run = True
    concat_corrected = pe.Node(MergeSeries(), name="concat_corrected")

    # fmt:off
    workflow.connect([
        (concat_blips, realign, [("out_file", "in_file")]),
        (realign, topup, [("out_file", "in_file")]),
        (fix_coeff, unwarp, [("out_coeff", "in_coeff")]),
        (realign, split_blips, [("out_file", "in_file")]),
        (split_blips, unwarp, [("out_files", "in_target")]),
        (readout_time, unwarp, [("readout_time", "ro_time"),
                                ("pe_direction", "pe_dir")]),
        (unwarp, outputnode, [("out_warp", "out_warps"),
                              ("out_field", "fmap")]),
        (unwarp, concat_corrected, [("out_corrected", "in_files")]),
        (concat_corrected, ref_average, [("out_file", "in_file")]),
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
