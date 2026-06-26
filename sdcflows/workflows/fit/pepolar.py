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

INPUT_FIELDS = ('metadata', 'in_data')
_PEPOLAR_DESC = """\
A *B<sub>0</sub>*-nonuniformity map (or *fieldmap*) was estimated based on two (or more)
echo-planar imaging (EPI) references """


def init_topup_wf(
    grid_reference=0,
    use_metadata_estimates=False,
    fallback_total_readout_time=None,
    omp_nthreads=1,
    sloppy=False,
    debug=False,
    name='pepolar_estimate_wf',
    topup_config=None,
    max_vols_per_pe=3,
    **kwargs,
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
    topup_config : :obj:`str`
        Path to custom topup config file.

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
    xfms : :obj:`list` of :obj:`str`
        The path(s) of the per-volume head-motion estimates.
    out_warps : :obj:`list` of :obj:`str`
        The path(s) of the per-volume displacement fields.
    jacobians : :obj:`list` of :obj:`str`
        The path(s) of the Jacobian determinant maps.
    method: :obj:`str`
        Short description of the estimation method that was run.
    topup_config : :obj:`str`
        The path of the config file used for TOPUP.

    """
    from nipype.interfaces.fsl.epi import TOPUP
    from niworkflows.interfaces.images import RobustAverage
    from niworkflows.interfaces.nibabel import MergeSeries, ReorientImage

    from ...interfaces.bspline import TOPUPCoeffReorient
    from ...interfaces.epi import GetReadoutTime, SelectPEVolumes, SortPEBlips
    from ...interfaces.utils import ReorientImageAndMetadata, UniformGrid
    from ...utils.misc import front as _front
    from ..ancillary import init_brainextraction_wf

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
{_PEPOLAR_DESC} with `topup` (@topup; FSL {TOPUP().version}).
"""

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'fmap',
                'fmap_ref',
                'fmap_coeff',
                'fmap_mask',
                'jacobians',
                'xfms',
                'out_warps',
                'method',
                'topup_config',
            ]
        ),
        name='outputnode',
    )
    outputnode.inputs.method = 'PEB/PEPOLAR (phase-encoding based / PE-POLARity)'

    # Calculate the total readout time of each run
    readout_time = pe.MapNode(
        GetReadoutTime(
            use_estimate=use_metadata_estimates,
        ),
        name='readout_time',
        iterfield=['metadata', 'in_file'],
        run_without_submitting=True,
    )
    if fallback_total_readout_time is not None:
        readout_time.inputs.fallback = fallback_total_readout_time
    # Cap volumes per PE direction so topup isn't overwhelmed (FSL: ~3 spares/dir)
    # https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/users_guide/index.html
    select_volumes = pe.Node(
        SelectPEVolumes(max_vols_per_pe=max_vols_per_pe), name='select_volumes'
    )
    # Regrid all to the reference (grid_reference=0 means first averaged run)
    regrid = pe.Node(UniformGrid(reference=grid_reference), name='regrid')
    # Sort PE blips to ensure reproducibility
    sort_pe_blips = pe.Node(SortPEBlips(), name='sort_pe_blips', run_without_submitting=True)
    # Merge into one 4D file
    concat_blips = pe.Node(MergeSeries(affine_tolerance=1e-4), name='concat_blips')
    # Feed the input images in LAS orientation, so FSL does not run funky reorientations
    to_las = pe.Node(ReorientImageAndMetadata(target_orientation='LAS'), name='to_las')
    # TOPUP will jointly estimate head motion and susceptibility distortion using the
    # default configuration files.
    topup = pe.Node(TOPUP(), name='topup')
    # "Generalize" topup coefficients and store them in a spatially-correct NIfTI file
    fix_coeff = pe.Node(TOPUPCoeffReorient(), name='fix_coeff', run_without_submitting=True)

    # Average the output
    ref_average = pe.Node(RobustAverage(num_threads=omp_nthreads), name='ref_average')

    # Sophisticated brain extraction of fMRIPrep
    brainextraction_wf = init_brainextraction_wf()

    workflow.connect([
        (inputnode, readout_time, [("in_data", "in_file"),
                                   ("metadata", "metadata")]),
        (inputnode, select_volumes, [("in_data", "in_data")]),
        (readout_time, select_volumes, [("pe_dir_fsl", "pe_dirs_fsl"),
                                        ("readout_time", "readout_times")]),
        (select_volumes, regrid, [("out_data", "in_data")]),
        (select_volumes, sort_pe_blips, [("pe_dirs_fsl", "pe_dirs_fsl"),
                                         ("readout_times", "readout_times")]),
        (regrid, sort_pe_blips, [("out_data", "in_data")]),
        (sort_pe_blips, topup, [("readout_times", "readout_times")]),
        (regrid, fix_coeff, [("reference", "fmap_ref")]),
        (sort_pe_blips, concat_blips, [("out_data", "in_files")]),
        (concat_blips, to_las, [("out_file", "in_file")]),
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
    ])  # fmt: skip

    # When no custom config is given, select the fastest appropriate config at runtime
    if topup_config:
        topup.inputs.config = str(topup_config)
        workflow.__desc__ += ' A custom `topup` configuration file was used.'
        outputnode.inputs.topup_config = str(topup_config)
    else:
        select_config = pe.Node(
            niu.Function(function=_select_topup_config, output_names=['config']),
            name='select_config',
            run_without_submitting=True,
        )
        select_config.inputs.sloppy = sloppy
        workflow.connect([
            (to_las, select_config, [('out_file', 'in_file')]),
            (select_config, topup, [('config', 'config')]),
            (select_config, outputnode, [('config', 'topup_config')]),
        ])  # fmt: skip

    if not debug:
        # Roll orientation back to original
        from_las = pe.Node(ReorientImage(), name='from_las')
        from_las_fmap = pe.Node(ReorientImage(), name='from_las_fmap')
        # fmt: off
        workflow.connect([
            (regrid, from_las, [("reference", "target_file")]),
            (regrid, from_las_fmap, [("reference", "target_file")]),
            (topup, from_las, [("out_corrected", "in_file")]),
            (from_las, ref_average, [("out_file", "in_file")]),
            (topup, from_las_fmap, [("out_field", "in_file")]),
            (topup, outputnode, [("out_warps", "out_warps")]),
            (from_las_fmap, outputnode, [("out_file", "fmap")]),
        ])
        # fmt: on
        return workflow

    from sdcflows.interfaces.bspline import ApplyCoeffsField

    unwarp = pe.Node(ApplyCoeffsField(jacobian=True), name='unwarp')
    unwarp.interface._always_run = True
    concat_corrected = pe.Node(MergeSeries(), name='concat_corrected')
    convert_xfms = pe.Node(
        niu.Function(function=_topup_mats_to_world, output_names=['out_xfms']),
        name='convert_xfms',
        run_without_submitting=True,
    )

    # fmt:off
    workflow.connect([
        (concat_blips, unwarp, [("out_file", "in_data")]),
        (fix_coeff, unwarp, [("out_coeff", "in_coeff")]),
        (topup, convert_xfms, [("out_mats", "in_mats")]),
        (to_las, convert_xfms, [("out_file", "in_reference")]),
        (convert_xfms, unwarp, [("out_xfms", "in_xfms")]),
        (sort_pe_blips, unwarp, [("readout_times", "ro_time"),
                                 ("pe_dirs", "pe_dir")]),
        (unwarp, outputnode, [("out_field", "fmap")]),
        (unwarp, concat_corrected, [("out_corrected", "in_files")]),
        (concat_corrected, ref_average, [("out_file", "in_file")]),
    ])
    # fmt:on

    return workflow


def init_3dQwarp_wf(omp_nthreads=1, debug=False, name='pepolar_estimate_wf'):
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
    from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf
    from niworkflows.interfaces.fixes import (
        FixHeaderApplyTransforms as ApplyTransforms,
    )
    from niworkflows.interfaces.fixes import (
        FixHeaderRegistration as Registration,
    )
    from niworkflows.interfaces.freesurfer import StructuralReference
    from niworkflows.interfaces.header import CopyHeader

    from ...interfaces.utils import ConvertWarp, Flatten
    from ...utils.misc import front as _front
    from ...utils.misc import last as _last

    workflow = Workflow(name=name)
    afni_ver = ''.join(f'{v:02d}' for v in afni.Info().version() or [])
    workflow.__desc__ = f'{_PEPOLAR_DESC} with `3dQwarp` (@afni; AFNI {afni_ver}).'

    inputnode = pe.Node(niu.IdentityInterface(fields=['in_data', 'metadata']), name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['fmap', 'fmap_ref']), name='outputnode')

    flatten = pe.Node(Flatten(), name='flatten')
    sort_pe = pe.Node(
        niu.Function(function=_sorted_pe, output_names=['sorted', 'qwarp_args']),
        name='sort_pe',
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
            out_file='template.nii.gz',
        ),
        name='merge_pes',
        iterfield=['in_files'],
    )

    pe0_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads, name='pe0_wf')
    pe1_wf = init_enhance_and_skullstrip_bold_wf(omp_nthreads=omp_nthreads, name='pe1_wf')

    align_pes = pe.Node(
        Registration(
            from_file=data.load('translation_rigid.json'),
            output_warped_image=True,
        ),
        name='align_pes',
        n_procs=omp_nthreads,
    )

    qwarp = pe.Node(
        afni.QwarpPlusMinus(
            blur=[-1, -1],
            environ={'OMP_NUM_THREADS': f'{min(omp_nthreads, 4)}'},
            minpatch=9,
            nopadWARP=True,
            noweight=True,
            pblur=[0.05, 0.05],
        ),
        name='qwarp',
        n_procs=min(omp_nthreads, 4),
    )

    to_ants = pe.Node(ConvertWarp(), name='to_ants', mem_gb=0.01)

    cphdr_warp = pe.Node(CopyHeader(), name='cphdr_warp', mem_gb=0.01)

    unwarp_reference = pe.Node(
        ApplyTransforms(
            dimension=3,
            float=True,
            interpolation='LanczosWindowedSinc',
        ),
        name='unwarp_reference',
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


def _select_topup_config(in_file, sloppy=False):
    """
    Pick an appropriate ``topup`` configuration for the input grid.

    TOPUP's ``--subsamp`` schedule collapses a ``s x s x s`` neighborhood into a
    single voxel at each level, so *every* image dimension must be divisible by the
    subsampling factor ``s`` used at that level (otherwise TOPUP errors out or
    segfaults). Previously, the workaround was zero-padding the troublesome dimensions
    to an even number of slices; however, later TOPUP releases conceded this was not ideal
    and now recommend using the config file that best matches the grid.

    The three stock configs differ **only** in two parameters; every other setting is identical:

    +----------------+---------------------------+-----------------------------+
    | config         | ``--subsamp``             | ``--lambda``                |
    |                | (max factor → constraint) | (regularization weight)     |
    +================+===========================+=============================+
    | ``b02b0_1``    | ``1,1,1,1,1,1,1,1,1``     | ``0.0005, 0.0001, ...``     |
    |                | (no subsampling; any dim) |                             |
    +----------------+---------------------------+-----------------------------+
    | ``b02b0_2``    | ``2,2,2,2,2,1,1,1,1``     | ``0.005, 0.001, ...``       |
    |                | (all dims divisible by 2) |                             |
    +----------------+---------------------------+-----------------------------+
    | ``b02b0_4``    | ``4,4,2,2,2,1,1,1,1``     | ``0.035, 0.006, ...``       |
    |                | (all dims divisible by 4) |                             |
    +----------------+---------------------------+-----------------------------+

    Coarser subsampling needs a heavier initial regularization weight (hence the
    larger leading ``--lambda`` values), which is why each tier ships its own
    ``--lambda`` schedule rather than reusing a single one. Aggressiveness (and
    therefore speed) increases down the table, at the cost of a stricter
    divisibility requirement, so we pick the largest factor the grid allows:

    * all dimensions divisible by four → ``b02b0_4.cnf`` (fastest)
    * all dimensions even → ``b02b0_2.cnf``
    * otherwise (any odd dimension) → ``b02b0_1.cnf`` (always safe)

    When ``sloppy`` is ``True``, the corresponding ``_quick`` variant is returned.
    The ``_quick`` configs are 3-level versions of the above
    (``--subsamp`` = ``1,1,1`` / ``2,2,2`` / ``4,2,1`` respectively) that keep the
    same per-tier ``--lambda`` tiering and estimate motion only in the first two
    levels (``--estmov=1,1,0``) to trade accuracy for speed in debug runs.

    """
    import nibabel as nb

    from sdcflows.data import load as load_data

    dims = nb.load(in_file).shape[:3]
    if all(d % 4 == 0 for d in dims):
        tier = '4'
    elif all(d % 2 == 0 for d in dims):
        tier = '2'
    else:
        tier = '1'

    return str(load_data(f'flirtsch/b02b0_{tier}{"_quick" * sloppy}.cnf'))


def _topup_mats_to_world(in_mats, in_reference):
    """
    Convert TOPUP's per-volume FSL motion matrices to world (RAS) affines.

    ``in_reference`` MUST be the image TOPUP realigned (the LAS-reoriented input):
    FSL matrices live in the voxel frame they were estimated in, so the same matrix
    yields a different physical transform under a different orientation. The resulting
    RAS-to-RAS affines are orientation-agnostic.
    """
    import nitransforms as nt

    return [
        nt.linear.load(mat, fmt='fsl', reference=in_reference, moving=in_reference).matrix.tolist()
        for mat in in_mats
    ]


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

    ref_pe = inlist[0][1]['PhaseEncodingDirection']
    for d, m in inlist[1:]:
        pe = m['PhaseEncodingDirection']
        if pe == ref_pe:
            out_ref.append(d)
        elif pe[0] == ref_pe[0]:
            out_opp.append(d)
        else:
            raise ValueError('Cannot handle orthogonal PE encodings.')

    return (
        [out_ref, out_opp],
        {'i': '-noYdis -noZdis', 'j': '-noXdis -noZdis', 'k': '-noXdis -noYdis'}[ref_pe[0]],
    )
