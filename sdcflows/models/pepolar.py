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

_PEPOLAR_DESC = """\
A *B<sub>0</sub>*-nonuniformity map (or *fieldmap*) was estimated based on two (or more)
echo-planar imaging (EPI) references """


def init_topup_wf(omp_nthreads=1, debug=False, name="pepolar_estimate_wf"):
    """
    Create the PEPOLAR field estimation workflow based on FSL's ``topup``.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.models.pepolar import init_3dQwarp_wf
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
        A list of EPI files that will be fed into TOPUP.
    metadata : :obj:`list` of :obj:`dict`
        A list of dictionaries containing the metadata corresponding to each file
        in ``in_data``.

    Outputs
    -------
    fieldmap : :obj:`str`
        The path of the estimated fieldmap.
    corrected : :obj:`str`
        The path of an unwarped conversion of files in ``in_data``.

    """
    from nipype.interfaces.fsl.epi import TOPUP
    from niworkflows.interfaces.nibabel import MergeSeries
    from sdcflows.interfaces.fmap import get_trt

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
{_PEPOLAR_DESC} with `topup` @topup (FSL {'.'.join(TOPUP().version())}).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["metadata", "in_data"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "corrected",
                "fieldmap",
                "coefficients",
                "jacobians",
                "xfms",
                "out_warps",
            ]
        ),
        name="outputnode",
    )

    concat_blips = pe.Node(MergeSeries(), name="concat_blips")
    readout_time = pe.MapNode(
        niu.Function(input_names=["in_meta", "in_file"], function=get_trt),
        name="readout_time",
        iterfield=["in_meta", "in_file"],
        run_without_submitting=True,
    )

    topup = pe.Node(
        TOPUP(
            config=_pkg_fname("sdcflows", f"data/flirtsch/b02b0{'_quick' * debug}.cnf")
        ),
        name="topup",
    )

    # fmt: off
    workflow.connect([
        (inputnode, concat_blips, [("in_data", "in_files")]),
        (inputnode, readout_time, [("in_data", "in_file"),
                                   ("metadata", "in_meta")]),
        (inputnode, topup, [(("metadata", _pe2fsl), "encoding_direction")]),
        (readout_time, topup, [("out", "readout_times")]),
        (concat_blips, topup, [("out_file", "in_file")]),
        (topup, outputnode, [("out_corrected", "corrected"),
                             ("out_field", "fieldmap"),
                             ("out_fieldcoef", "coefficients"),
                             ("out_jacs", "jacobians"),
                             ("out_mats", "xfms"),
                             ("out_warps", "out_warps")]),
    ])
    # fmt: on

    return workflow


def init_3dQwarp_wf(pe_dir, omp_nthreads=1, name="pepolar_estimate_wf"):
    """
    Create the PEPOLAR field estimation workflow based on AFNI's ``3dQwarp``.

    This workflow takes in two EPI files that MUST have opposed
    :abbr:`PE (phase-encoding)` direction.
    Therefore, EPIs with orthogonal PE directions are not supported.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.models.pepolar import init_3dQwarp_wf
            wf = init_3dQwarp_wf()

    Parameters
    ----------
    pe_dir : :obj:`str`
        PE direction (BIDS compatible)
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
    fieldmap : :obj:`str`
        The path of the estimated fieldmap.
    corrected : :obj:`str`
        The path of an unwarped conversion of the first element of ``in_data``.

    """
    from nipype.interfaces import afni
    from niworkflows.interfaces import CopyHeader
    from niworkflows.interfaces.registration import ANTSApplyTransformsRPT

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""{_PEPOLAR_DESC} \
with `3dQwarp` @afni (AFNI {''.join(['%02d' % v for v in afni.Info().version() or []])}).
"""

    inputnode = pe.Node(niu.IdentityInterface(fields=["in_data"]), name="inputnode")

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fieldmap", "corrected"]), name="outputnode"
    )

    qwarp = pe.Node(
        afni.QwarpPlusMinus(
            args={
                "i": "-noYdis -noZdis",
                "j": "-noXdis -noZdis",
                "k": "-noXdis -noYdis",
            }[pe_dir[0]],
            blur=[-1, -1],
            environ={"OMP_NUM_THREADS": f"{omp_nthreads}"},
            minpatch=9,
            nopadWARP=True,
            noweight=True,
            pblur=[0.05, 0.05],
        ),
        name="qwarp",
        n_procs=omp_nthreads,
    )

    to_ants = pe.Node(niu.Function(function=_fix_hdr), name="to_ants", mem_gb=0.01)

    cphdr_warp = pe.Node(CopyHeader(), name="cphdr_warp", mem_gb=0.01)

    unwarp_reference = pe.Node(
        ANTSApplyTransformsRPT(
            dimension=3,
            generate_report=False,
            float=True,
            interpolation="LanczosWindowedSinc",
        ),
        name="unwarp_reference",
    )

    # fmt: off
    workflow.connect([
        (inputnode, qwarp, [(("in_data", _front), "in_file"),
                            (("in_data", _last), "base_file")]),
        (inputnode, cphdr_warp, [(("in_data", _front), "hdr_file")]),
        (qwarp, cphdr_warp, [("source_warp", "in_file")]),
        (cphdr_warp, to_ants, [("out_file", "in_file")]),
        (to_ants, unwarp_reference, [("out", "transforms")]),
        (inputnode, unwarp_reference, [("in_reference", "reference_image"),
                                       ("in_reference", "input_image")]),
        (unwarp_reference, outputnode, [("output_image", "corrected")]),
        (to_ants, outputnode, [("out", "fieldmap")]),
    ])
    # fmt: on
    return workflow


def _fix_hdr(in_file, newpath=None):
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    hdr = nii.header.copy()
    hdr.set_data_dtype("<f4")
    hdr.set_intent("vector", (), "")
    out_file = fname_presuffix(in_file, "_warpfield", newpath=newpath)
    nb.Nifti1Image(nii.get_fdata(dtype="float32"), nii.affine, hdr).to_filename(
        out_file
    )
    return out_file


def _pe2fsl(metadata):
    """Convert ijk notation to xyz."""
    return [chr(ord(m["PhaseEncodingDirection"]) + 15) for m in metadata]


def _front(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def _last(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[-1]
    return inlist
