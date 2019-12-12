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

import pkg_resources as pkgr

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces import CopyHeader
from niworkflows.interfaces.freesurfer import StructuralReference
from niworkflows.func.util import init_enhance_and_skullstrip_bold_wf

from nipype.pipeline import engine as pe
from nipype.interfaces import afni, ants, fsl, utility as niu

def init_topup_wf(omp_nthreads=1, matched_pe=False, name="topup_wf"):
    """
    Create the PE-Polar field estimation workflow.

    This workflow takes in a set of EPI files with opposite phase encoding
    direction than the target file and calculates a fieldmap (off-resonance field) in Hz that can be fed into the ``fmap`` workflows

    This procedure works if there is only one ``_epi`` file is present
    (as long as it has the opposite phase encoding direction to the target
    file). The target file will be used to estimate the field distortion.
    However, if there is another ``_epi`` file present with a matching
    phase encoding direction to the target it will be used instead.

    Currently, different phase encoding directions in the target file and the
    ``_epi`` file(s) (for example, ``i`` and ``j``) is not supported.

    The off-resonance field is estimated using FSL's ``Topup``. Topup also calculates undistorted versions of the inputted EPI files which can be used as magnitute input in the ``fmap`` worklows.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.topup import init_topup_wf
            wf = init_topup_wf()

    Parameters
    ----------
    matched_pe : bool
        Whether the input ``fmaps_epi`` will contain images with matched
        PE blips or not. Please use :func:`sdcflows.workflows.topup.check_pes`
        to determine whether they exist or not.
    name : str
        Name for this workflow
    omp_nthreads : int
        Parallelize internal tasks across the number of CPUs given by this option.

    Inputs
    ------
    fmaps_epi : list of tuple(pathlike, str)
        The list of EPI images that will be used in PE-Polar correction, and
        their corresponding ``PhaseEncodingDirection`` metadata.
        The workflow will use the ``epi_pe_dir`` input to separate out those
        EPI acquisitions with opposed PE blips and those with matched PE blips
        (the latter could be none, and ``in_reference_brain`` would then be
        used). The workflow raises a ``ValueError`` when no images with
        opposed PE blips are found.
    epi_pe_dir : str
        The baseline PE direction.
    epi_trt: float
        Total readout time of the EPI (should be identical to TRT of fmaps_epi)
    matched_pe_dir : str
        Phase encoding direction of matching fmap EPI
    opposed_pe_dir : str
        Phase encoding direction of opposed fmap EPI
    in_reference_brain : pathlike
        The skullstripped baseline reference image (must correspond to ``epi_pe_dir``).

    Outputs
    -------
    fieldmap : pathlike
        Topup estimated fieldmap (Hz)
    magnitude : pathlike
        The ``fmaps_epi`` after unwarping and skullstripping
    """

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A B0-nonuniformity map (or *fieldmap*) was estimated based on two (or more)
echo-planar imaging (EPI) references with opposing phase-encoding
directions, with `Topup`).
"""

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['fmaps_epi', 'epi_pe_dir', 'epi_trt', 'in_reference_brain', 'matched_pe_dir', 'opposed_pe_dir']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fieldmap', 'magnitude']),
        name='outputnode')

    prepare_epi_wf = init_prepare_epi_wf(omp_nthreads=omp_nthreads,
                                         matched_pe=matched_pe,
                                         name="prepare_epi_wf")

    epi_merge2list = pe.Node(niu.Merge(2), name='epi_merge2list')
    pedirs_merge2list = pe.Node(niu.Merge(2), name='pedir_merge2list')
    trt_merge2list = pe.Node(niu.Merge(2), name='trt_merge2list')

    merger = pe.Node(fsl.Merge(dimension='t'), name='merger')

    topup = pe.Node(fsl.TOPUP(), name='topup')

    workflow.connect([
        (inputnode, prepare_epi_wf, [
            ('fmaps_epi', 'inputnode.maps_pe'),
            ('epi_pe_dir', 'inputnode.epi_pe'),
            ('in_reference_brain', 'inputnode.ref_brain')]),
        (prepare_epi_wf, epi_merge2list, [
            ('outputnode.matched_pe', 'in1'),
            ('outputnode.opposed_pe', 'in2')]),
        (epi_merge2list, merger, [
            ('out', 'in_files')]),
        (merger, topup, [
            ('merged_file', 'in_file')]),
        (inputnode, pedirs_merge2list, [
            (('matched_pe_dir', _get_pedir_topup), 'in1'),
            (('opposed_pe_dir',_get_pedir_topup), 'in2')]),
        (pedirs_merge2list, topup, [
            ('out', 'encoding_direction')]),
        (inputnode, trt_merge2list, [
            ('epi_trt', 'in1'),
            ('epi_trt', 'in2')]),
        (trt_merge2list, topup, [
            ('out', 'readout_times')]),
        (topup, outputnode, [
            ('out_field', 'fieldmap'),
            ('out_corrected', 'magnitude')]),
    ])

    return workflow


def init_prepare_epi_wf(omp_nthreads, matched_pe=False,
                        name="prepare_epi_wf"):
    """
    Prepare opposed-PE EPI images for PE-POLAR SDC.

    This workflow takes in a set of EPI files and returns two 3D volumes with
    matching and opposed PE directions, ready to be used in field distortion
    estimation.

    The procedure involves: estimating a robust template using FreeSurfer's
    ``mri_robust_template``, bias field correction using ANTs ``N4BiasFieldCorrection``
    and AFNI ``3dUnifize``, skullstripping using FSL BET and AFNI ``3dAutomask``,
    and rigid coregistration to the reference using ANTs.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.topup import init_prepare_epi_wf
            wf = init_prepare_epi_wf(omp_nthreads=8)

    Parameters
    ----------
    matched_pe : bool
        Whether the input ``fmaps_epi`` will contain images with matched
        PE blips or not. Please use :func:`sdcflows.workflows.topup.check_pes`
        to determine whether they exist or not.
    name : str
        Name for this workflow
    omp_nthreads : int
        Parallelize internal tasks across the number of CPUs given by this option.

    Inputs
    ------
    epi_pe : str
        Phase-encoding direction of the EPI image to be corrected.
    maps_pe : list of tuple(pathlike, str)
        list of 3D or 4D NIfTI images
    ref_brain
        coregistration reference (skullstripped and bias field corrected)

    Outputs
    -------
    opposed_pe : pathlike
        single 3D NIfTI file
    matched_pe : pathlike
        single 3D NIfTI file

    """
    inputnode = pe.Node(niu.IdentityInterface(fields=['epi_pe', 'maps_pe', 'ref_brain']),
                        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(fields=['opposed_pe', 'matched_pe']),
                         name='outputnode')

    ants_settings = pkgr.resource_filename('sdcflows',
                                           'data/translation_rigid.json')

    split = pe.Node(niu.Function(function=_split_epi_lists), name='split')

    merge_op = pe.Node(
        StructuralReference(auto_detect_sensitivity=True,
                            initial_timepoint=1,
                            fixed_timepoint=True,  # Align to first image
                            intensity_scaling=True,
                            # 7-DOF (rigid + intensity)
                            no_iteration=True,
                            subsample_threshold=200,
                            out_file='template.nii.gz'),
        name='merge_op')

    ref_op_wf = init_enhance_and_skullstrip_bold_wf(
        omp_nthreads=omp_nthreads, name='ref_op_wf')

    op2ref_reg = pe.Node(ants.Registration(
        from_file=ants_settings, output_warped_image=True),
        name='op2ref_reg', n_procs=omp_nthreads)

    workflow = Workflow(name=name)
    workflow.connect([
        (inputnode, split, [('maps_pe', 'in_files'),
                            ('epi_pe', 'pe_dir')]),
        (split, merge_op, [(('out', _front), 'in_files')]),
        (merge_op, ref_op_wf, [('out_file', 'inputnode.in_file')]),
        (ref_op_wf, op2ref_reg, [
            ('outputnode.skull_stripped_file', 'moving_image')]),
        (inputnode, op2ref_reg, [('ref_brain', 'fixed_image')]),
        (op2ref_reg, outputnode, [('warped_image', 'opposed_pe')]),
    ])

    if not matched_pe:
        workflow.connect([
            (inputnode, outputnode, [('ref_brain', 'matched_pe')]),
        ])
        return workflow

    merge_ma = pe.Node(
        StructuralReference(auto_detect_sensitivity=True,
                            initial_timepoint=1,
                            fixed_timepoint=True,  # Align to first image
                            intensity_scaling=True,
                            # 7-DOF (rigid + intensity)
                            no_iteration=True,
                            subsample_threshold=200,
                            out_file='template.nii.gz'),
        name='merge_ma')

    ref_ma_wf = init_enhance_and_skullstrip_bold_wf(
        omp_nthreads=omp_nthreads, name='ref_ma_wf')

    ma2ref_reg = pe.Node(ants.Registration(
        from_file=ants_settings, output_warped_image=True),
        name='ma2ref_reg', n_procs=omp_nthreads)

    workflow.connect([
        (split, merge_ma, [(('out', _last), 'in_files')]),
        (merge_ma, ref_ma_wf, [('out_file', 'inputnode.in_file')]),
        (ref_ma_wf, ma2ref_reg, [
            ('outputnode.skull_stripped_file', 'moving_image')]),
        (inputnode, ma2ref_reg, [('ref_brain', 'fixed_image')]),
        (ma2ref_reg, outputnode, [('warped_image', 'matched_pe')]),
    ])
    return workflow


def _front(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def _last(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[-1]
    return inlist


def check_pes(epi_fmaps, pe_dir):
    """Check whether there are images with matched PE."""
    opposed_pe = False
    matched_pe = False
    matched_pe_dir = pe_dir
    opposed_pe_dir = ''

    for _, fmap_pe in epi_fmaps:
        if fmap_pe == pe_dir:
            matched_pe = True
        elif fmap_pe[0] == pe_dir[0]:
            opposed_pe = True
            opposed_pe_dir = fmap_pe

    if not opposed_pe:
        raise ValueError("""\
None of the discovered fieldmaps has the right phase encoding direction. \
This is possibly a problem with metadata. If not, rerun with \
``--ignore fieldmaps`` to skip the distortion correction step.""")

    return matched_pe, matched_pe_dir, opposed_pe_dir


def _split_epi_lists(in_files, pe_dir, max_trs=50):
    """
    Split input EPIs and generate an output list of PEs.

    Inputs
    ------
    in_files : list of ``BIDSFile``s
        The EPI images that will be pooled into field estimation.
    pe_dir : str
        The phase-encoding direction of the IntendedFor EPI scan.
    max_trs : int
        Index of frame after which all volumes will be discarded
        from the input EPI images.

    """
    from os import path as op
    import nibabel as nb

    matched_pe = []
    opposed_pe = []

    for i, (epi_path, epi_pe) in enumerate(in_files):
        if epi_pe[0] == pe_dir[0]:
            img = nb.load(epi_path)
            if len(img.shape) == 3:
                splitnii = [img]
            else:
                splitnii = nb.four_to_three(img.slicer[:, :, :, :max_trs])

            for j, nii in enumerate(splitnii):
                out_name = op.abspath(
                    'dir-%s_tstep-%03d_pe-%03d.nii.gz' % (epi_pe, i, j))
                nii.to_filename(out_name)

                if epi_pe == pe_dir:
                    matched_pe.append(out_name)
                else:
                    opposed_pe.append(out_name)

    if matched_pe:
        return [opposed_pe, matched_pe]

    return [opposed_pe]

def _get_pedir_topup(in_pe):
    return in_pe.replace('i', 'x').replace('j', 'y').replace('k', 'z')