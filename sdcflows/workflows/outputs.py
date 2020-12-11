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
    *,
    output_dir,
    fmap_type,
    bids_fmap_id=None,
    custom_entities=None,
    name="fmap_reports_wf",
):
    """
    Set up a battery of datasinks to store reports in the right location.

    Parameters
    ----------
    fmap_type : :obj:`str`
        The fieldmap estimator type.
    output_dir : :obj:`str`
        Directory in which to save derivatives
    bids_fmap_id : :obj:`str`
        Sets the ``B0FieldIdentifier`` metadata into the outputs.
    custom_entities : :obj:`dict`
        Define extra entities that will be written out in filenames.
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
    if bids_fmap_id:
        custom_entities["fmapid"] = bids_fmap_id.replace("_", "")

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
    output_dir : :obj:`str`
        Directory in which to save derivatives
    bids_fmap_id : :obj:`str`
        Sets the ``B0FieldIdentifier`` metadata into the outputs.
    custom_entities : :obj:`dict`
        Define extra entities that will be written out in filenames.
    name : :obj:`str`
        Workflow name (default: ``"fmap_derivatives_wf"``)
    write_coeff : :obj:`bool`
        Build the workflow path to map coefficients into target space.

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
    if bids_fmap_id:
        custom_entities["fmapid"] = bids_fmap_id.replace("_", "")

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


def _gendesc(infiles):
    """
    Generate a desc entity value.

    Examples
    --------
    >>> _gendesc("f")
    'coeff'

    >>> _gendesc(list("ab"))
    ['coeff0', 'coeff1']

    """
    if isinstance(infiles, (str, bytes)):
        infiles = [infiles]

    if len(infiles) == 1:
        return "coeff"

    return [f"coeff{i}" for i, _ in enumerate(infiles)]


def _getname(infile):
    """
    Get file names only.

    Examples
    --------
    >>> _getname("drop/path/filename.txt")
    'filename.txt'

    >>> _getname(["drop/path/filename.txt", "other/path/filename2.txt"])
    ['filename.txt', 'filename2.txt']

    """
    from pathlib import Path

    if isinstance(infile, (list, tuple)):
        return [Path(f).name for f in infile]
    return Path(infile).name


def _getsourcetype(infiles):
    """
    Determine the type of fieldmap estimation strategy.

    Example
    -------
    >>> _getsourcetype(["path/some_epi.nii.gz"])
    'epi'

    >>> _getsourcetype(["path/some_notepi.nii.gz"])
    'magnitude'

    """
    from pathlib import Path

    fname = Path(infiles[0]).name
    return "epi" if fname.endswith(("_epi.nii.gz", "_epi.nii")) else "magnitude"


def _selectintent(metadata):
    """
    Extract the IntendedFor metadata.

    Example
    -------
    >>> _selectintent({})
    []

    >>> _selectintent({"IntendedFor": "just/one/file.txt"})
    ['just/one/file.txt']

    >>> _selectintent({"IntendedFor": ["file2.txt", "file1.txt"]})
    ['file1.txt', 'file2.txt']

    >>> _selectintent([{"IntendedFor": "just/one/file.txt"}] * 2)
    ['just/one/file.txt']

    >>> _selectintent([
    ...     {"IntendedFor": "just/one/file.txt"},
    ...     {"IntendedFor": ["file2.txt", "file1.txt"]},
    ... ])
    ['file1.txt', 'file2.txt', 'just/one/file.txt']

    """
    from bids.utils import listify

    return sorted(
        set([el for m in listify(metadata) for el in listify(m.get("IntendedFor", []))])
    )
