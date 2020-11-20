# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Writing out outputs."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.interfaces.bids import DerivativesDataSink as _DDS


class DerivativesDataSink(_DDS):
    """Overload the ``out_path_base`` setting."""

    out_path_base = "sdcflows"


del _DDS


def init_fmap_reports_wf(
    *, output_dir, fmap_type, custom_entities=None, name="fmap_reports_wf",
):
    """
    Set up a battery of datasinks to store reports in the right location.

    Parameters
    ----------
    fmap_type : :obj:`str`
        The fieldmap estimator type.
    custom_entities : :obj:`dict`
        Define extra entities that will be written out in filenames.
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: ``"fmap_reports_wf"``)

    Inputs
    ------
    source_files
        One or more fieldmap file(s) of the BIDS dataset that will serve for naming reference.
    fieldmap
        The preprocessed fieldmap, in its original space with Hz units.
    fmap_ref
        An anatomical reference (e.g., magnitude file)
    fmap_mask
        A brain mask in the fieldmap's space.

    """
    from ..interfaces.reportlets import FieldmapReportlet

    custom_entities = custom_entities or {}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["source_files", "fieldmap", "fmap_ref", "fmap_mask"]
        ),
        name="inputnode",
    )

    rep = pe.Node(FieldmapReportlet(), "simple_report")
    rep.interface._always_run = True

    ds_fmap_report = pe.Node(
        DerivativesDataSink(
            base_directory=str(output_dir),
            datatype="figures",
            suffix="fieldmap",
            desc=fmap_type,
            dismiss_entities=("fmap",),
            allowed_entities=tuple(custom_entities.keys()),
        ),
        name="ds_fmap_report",
    )
    for k, v in custom_entities.items():
        setattr(ds_fmap_report.inputs, k, v)

    # fmt:off
    workflow.connect([
        (inputnode, rep, [("fieldmap", "fieldmap"),
                          ("fmap_ref", "reference"),
                          ("fmap_mask", "mask")]),
        (rep, ds_fmap_report, [("out_report", "in_file")]),
        (inputnode, ds_fmap_report, [("source_files", "source_file")]),

    ])
    # fmt:on

    return workflow


def init_fmap_derivatives_wf(
    *,
    output_dir,
    bids_fmap_id=None,
    custom_entities=None,
    name="fmap_derivatives_wf",
    write_coeff=False,
):
    """
    Set up datasinks to store derivatives in the right location.

    Parameters
    ----------
    bids_fmap_id : :obj:`str`
        Sets the ``B0FieldIdentifier`` metadata into the outputs.
    custom_entities : :obj:`dict`
        Define extra entities that will be written out in filenames.
    output_dir : :obj:`str`
        Directory in which to save derivatives
    name : :obj:`str`
        Workflow name (default: ``"fmap_derivatives_wf"``)

    Inputs
    ------
    source_files
        One or more fieldmap file(s) of the BIDS dataset that will serve for naming reference.
    fieldmap
        The preprocessed fieldmap, in its original space with Hz units.
    fmap_coeff
        Field coefficient(s) file(s)
    fmap_ref
        An anatomical reference (e.g., magnitude file)

    """
    custom_entities = custom_entities or {}

    workflow = pe.Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["source_files", "fieldmap", "fmap_coeff", "fmap_ref", "fmap_meta"]
        ),
        name="inputnode",
    )

    ds_reference = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            compress=True,
            suffix="fieldmap",
            dismiss_entities=("fmap",),
            allowed_entities=tuple(custom_entities.keys()),
        ),
        name="ds_reference",
    )

    ds_fieldmap = pe.Node(
        DerivativesDataSink(
            base_directory=output_dir,
            desc="preproc",
            suffix="fieldmap",
            compress=True,
            allowed_entities=tuple(custom_entities.keys()),
        ),
        name="ds_fieldmap",
    )
    ds_fieldmap.inputs.Units = "Hz"
    if bids_fmap_id:
        ds_fieldmap.inputs.B0FieldIdentifier = bids_fmap_id

    for k, v in custom_entities.items():
        setattr(ds_reference.inputs, k, v)
        setattr(ds_fieldmap.inputs, k, v)

    # fmt:off
    workflow.connect([
        (inputnode, ds_reference, [("source_files", "source_file"),
                                   ("fmap_ref", "in_file"),
                                   (("source_files", _getsourcetype), "desc")]),
        (inputnode, ds_fieldmap, [("source_files", "source_file"),
                                  ("fieldmap", "in_file"),
                                  ("source_files", "RawSources")]),
        (ds_reference, ds_fieldmap, [
            (("out_file", _getname), "AnatomicalReference"),
        ]),
        (inputnode, ds_fieldmap, [(("fmap_meta", _selectintent), "IntendedFor")]),
    ])
    # fmt:on

    if not write_coeff:
        return workflow

    ds_coeff = pe.MapNode(
        DerivativesDataSink(
            base_directory=output_dir,
            suffix="fieldmap",
            compress=True,
            allowed_entities=tuple(custom_entities.keys()),
        ),
        name="ds_coeff",
        iterfield=("in_file", "desc"),
    )

    gen_desc = pe.Node(niu.Function(function=_gendesc), name="gen_desc")

    for k, v in custom_entities.items():
        setattr(ds_coeff.inputs, k, v)

    # fmt:off
    workflow.connect([
        (inputnode, ds_coeff, [("source_files", "source_file"),
                               ("fmap_coeff", "in_file")]),
        (inputnode, gen_desc, [("fmap_coeff", "infiles")]),
        (gen_desc, ds_coeff, [("out", "desc")]),
        (ds_coeff, ds_fieldmap, [(("out_file", _getname), "AssociatedCoefficients")]),
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


def _getname(infile):
    from pathlib import Path

    if isinstance(infile, (list, tuple)):
        return [Path(f).name for f in infile]
    return Path(infile).name


def _getsourcetype(infiles):
    from pathlib import Path

    fname = Path(infiles[0]).name
    return "epi" if fname.endswith(("_epi.nii.gz", "_epi.nii")) else "magnitude"


def _selectintent(metadata):
    from bids.utils import listify

    return sorted(
        set([el for m in metadata for el in listify(m.get("IntendedFor", []))])
    )
