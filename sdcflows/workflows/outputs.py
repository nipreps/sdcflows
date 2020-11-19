# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Writing out outputs."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.interfaces.bids import DerivativesDataSink


def init_fmap_derivatives_wf(
    *,
    bids_root,
    output_dir,
    write_coeff=False,
    name="fmap_derivatives_wf",
):
    """
    Set up datasinks to store derivatives in the right location.

    Parameters
    ----------
    bids_root : :obj:`str`
        Root path of BIDS dataset
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: ``"fmap_derivatives_wf"``)

    Inputs
    ------
    source_file
        A fieldmap file of the BIDS dataset that will serve for naming reference.
    fieldmap
        The preprocessed fieldmap, in its original space with Hz units.
    fmap_coeff
        Field coefficient(s) file(s)
    fmap_ref
        An anatomical reference (e.g., magnitude file)

    """
    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "source_file",
                "fieldmap",
                "fmap_coeff",
                "fmap_ref",
            ]),
        name="inputnode"
    )

    ds_reference = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="reference",
            suffix="fieldmap",
            compress=True),
        name="ds_reference",
    )

    ds_fieldmap = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="preproc",
            suffix="fieldmap",
            compress=True),
        name="ds_fieldmap",
    )
    ds_fieldmap.inputs.Units = "Hz"

    # fmt:off
    workflow.connect([
        (inputnode, ds_reference, [("source_file", "source_file"),
                                   ("fmap_ref", "in_file")]),
        (inputnode, ds_fieldmap, [("source_file", "source_file"),
                                  ("fieldmap", "in_file")]),
    ])
    # fmt:on

    if not write_coeff:
        return workflow

    ds_coeff = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            suffix="fieldmap",
            compress=True),
        name="ds_coeff",
        iterfield=("in_file", "desc"),
    )

    gen_desc = pe.Node(niu.Function(function=_gendesc), name="gen_desc")

    # fmt:off
    workflow.connect([
        (inputnode, ds_coeff, [("source_file", "source_file"),
                               ("fmap_coeff", "in_file")]),
        (inputnode, gen_desc, [("fmap_coeff", "infiles")]),
        (gen_desc, ds_coeff, [("out", "desc")]),
    ])
    # fmt:on

    return workflow


def init_sdc_unwarp_report_wf(name="sdc_unwarp_report_wf", forcedsyn=False):
    """
    Save a reportlet showing how SDC unwarping performed.

    This workflow generates and saves a reportlet showing the effect of fieldmap
    unwarping a BOLD image.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.outputs import init_sdc_unwarp_report_wf
            wf = init_sdc_unwarp_report_wf()

    Parameters
    ----------
    name : str, optional
        Workflow name (default: ``sdc_unwarp_report_wf``)
    forcedsyn : bool, optional
        Whether SyN-SDC was forced.

    Inputs
    ------
    in_pre
        Reference image, before unwarping
    in_post
        Reference image, after unwarping
    in_seg
        Segmentation of preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF)
    in_xfm
        Affine transform from T1 space to BOLD space (ITK format)

    """
    from niworkflows.interfaces import SimpleBeforeAfter
    from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
    from niworkflows.utils.images import dseg_label as _dseg_label

    DEFAULT_MEMORY_MIN_GB = 0.01

    workflow = pe.Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["in_pre", "in_post", "in_seg", "in_xfm"]),
        name="inputnode",
    )

    map_seg = pe.Node(
        ApplyTransforms(dimension=3, float=True, interpolation="MultiLabel"),
        name="map_seg",
        mem_gb=0.3,
    )

    sel_wm = pe.Node(
        niu.Function(function=_dseg_label), name="sel_wm", mem_gb=DEFAULT_MEMORY_MIN_GB
    )
    sel_wm.inputs.label = 2

    bold_rpt = pe.Node(SimpleBeforeAfter(), name="bold_rpt", mem_gb=0.1)
    ds_report_sdc = pe.Node(
        DerivativesDataSink(
            desc=("sdc", "forcedsyn")[forcedsyn], suffix="bold", datatype="figures"
        ),
        name="ds_report_sdc",
        mem_gb=DEFAULT_MEMORY_MIN_GB,
        run_without_submitting=True,
    )

    # fmt: off
    workflow.connect([
        (inputnode, bold_rpt, [("in_post", "after"),
                               ("in_pre", "before")]),
        (bold_rpt, ds_report_sdc, [("out_report", "in_file")]),
        (inputnode, map_seg, [("in_post", "reference_image"),
                              ("in_seg", "input_image"),
                              ("in_xfm", "transforms")]),
        (map_seg, sel_wm, [("output_image", "in_seg")]),
        (sel_wm, bold_rpt, [("out", "wm_seg")]),
    ])
    # fmt: on

    return workflow


def _gendesc(infiles):
    if isinstance(infiles, (str, bytes)):
        infiles = [infiles]

    if len(infiles) == 1:
        return "coeff"

    return [f"coeff{i}" for i, _ in enumerate(infiles)]
