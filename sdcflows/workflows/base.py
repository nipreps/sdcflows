# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""SDC workflows coordination."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from nipype import logging

from niworkflows.engine.workflows import LiterateWorkflow as Workflow


LOGGER = logging.getLogger("nipype.workflow")
FMAP_PRIORITY = {
    "epi": 0,
    "fieldmap": 1,
    "phasediff": 2,
    "syn": 3,
}

DEFAULT_MEMORY_MIN_GB = 0.01


def init_sdc_estimate_wf(bids_fmaps, omp_nthreads=1, debug=False):
    """
    Build a :abbr:`SDC (susceptibility distortion correction)` workflow.

    This workflow implements the heuristics to choose an estimation
    methodology for :abbr:`SDC (susceptibility distortion correction)`.
    When no field map information is present within the BIDS inputs,
    the EXPERIMENTAL "fieldmap-less SyN" can be performed, using
    the ``--use-syn`` argument. When ``--force-syn`` is specified,
    then the "fieldmap-less SyN" is always executed and reported
    despite of other fieldmaps available with higher priority.
    In the latter case (some sort of fieldmap(s) is available and
    ``--force-syn`` is requested), then the :abbr:`SDC (susceptibility
    distortion correction)` method applied is that with the
    highest priority.

    Parameters
    ----------
    bids_fmaps : list of pybids dicts
        A list of dictionaries with the available fieldmaps
        (and their metadata using the key ``'metadata'`` for the
        case of :abbr:`PEPOLAR (Phase-Encoding POLARity)` fieldmaps).
    omp_nthreads : int
        Maximum number of threads an individual process may use
    debug : bool
        Enable debugging outputs

    Inputs
    ------
    epi_file
        A reference image calculated at a previous stage
    epi_brain
        Same as above, but brain-masked
    epi_mask
        Brain mask for the run
    t1w_brain
        T1w image, brain-masked, for the fieldmap-less SyN method
    std2anat_xfm
        Standard-to-T1w transform generated during spatial
        normalization (only for the fieldmap-less SyN method).

    Outputs
    -------
    epi_corrected
        The EPI scan reference after unwarping.
    epi_mask
        The corresponding new mask after unwarping
    epi_brain
        Brain-extracted, unwarped EPI scan reference
    out_warp
        The deformation field to unwarp the susceptibility distortions
    method : str
        Short description of the estimation method that was run.

    """
    workflow = Workflow(name="sdc_estimate_wf" if bids_fmaps else "sdc_bypass_wf")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fieldmap", "fmap_ref", "method"]),
        name="outputnode",
    )

    # No fieldmaps - forward inputs to outputs
    if not bids_fmaps:
        workflow.__postdesc__ = """\
Susceptibility distortion correction (SDC) was omitted.
"""
        outputnode.inputs.method = "None"
        outputnode.inputs.fieldmap = "identity"
        # fmt: off
        workflow.add_nodes([outputnode])
        # fmt: on
        return workflow

    workflow.__postdesc__ = """\
Based on the estimated susceptibility distortion, a corrected
EPI (echo-planar imaging) reference was calculated for a more
accurate co-registration with the anatomical reference.
"""

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["epi_file", "epi_brain", "epi_mask", "t1w_brain", "std2anat_xfm"]
        ),
        name="inputnode",
    )
    # PEPOLAR path
    if "epi" in bids_fmaps:
        from .fit.pepolar import init_3dQwarp_wf

        outputnode.inputs.method = "PEB/PEPOLAR (phase-encoding based / PE-POLARity)"

        in_data, metadata = zip(*bids_fmaps["epi"])
        estimate_wf = init_3dQwarp_wf(omp_nthreads=omp_nthreads)
        estimate_wf.inputs.inputnode.in_data = in_data
        estimate_wf.inputs.inputnode.metadata = metadata

    # FIELDMAP path
    elif "fieldmap" in bids_fmaps or "phasediff" in bids_fmaps:
        from .fit.fieldmap import init_fmap_wf

        if "fieldmap" in bids_fmaps:
            fmap = bids_fmaps["fieldmap"][0]
            outputnode.inputs.method = "FMB (fieldmap-based) - directly measured B0 map"
            estimate_wf = init_fmap_wf(omp_nthreads=omp_nthreads, mode="fieldmap")
            estimate_wf.inputs.inputnode.fieldmap = [m for m, _ in fmap["fieldmap"]]
        else:
            fmap = bids_fmaps["phasediff"][0]
            outputnode.inputs.method = "FMB (fieldmap-based) - phase-difference map"
            estimate_wf = init_fmap_wf(omp_nthreads=omp_nthreads)
            estimate_wf.inputs.inputnode.fieldmap = fmap["phases"]

        # set magnitude files (common for the three flavors)
        estimate_wf.inputs.inputnode.magnitude = [m for m, _ in fmap["magnitude"]]

    # FIELDMAP-less path
    elif "syn" in bids_fmaps:
        from .fit.syn import init_syn_sdc_wf

        outputnode.inputs.method = 'FLB ("fieldmap-less", SyN-based)'

        estimate_wf = init_syn_sdc_wf(omp_nthreads=omp_nthreads)

        # fmt: off
        workflow.connect([
            (inputnode, estimate_wf, [
                ("epi_file", "inputnode.in_reference"),
                ("epi_brain", "inputnode.in_reference_brain"),
                ("t1w_brain", "inputnode.t1w_brain"),
                ("std2anat_xfm", "inputnode.std2anat_xfm")]),
        ])
        # fmt: on
    else:
        raise ValueError("Unsupported field mapping strategy.")

    # fmt: off
    workflow.connect([
        (estimate_wf, outputnode, [
            ("outputnode.fmap", "fieldmap"),
            ("outputnode.fmap_ref", "reference")]),
    ])
    # fmt: on
    return workflow


def fieldmap_wrangler(layout, target_image, use_syn=False, force_syn=False):
    """Query the BIDSLayout for fieldmaps, and arrange them for the orchestration workflow."""
    from collections import defaultdict

    fmap_bids = layout.get_fieldmap(target_image, return_list=True)
    fieldmaps = defaultdict(list)
    for fmap in fmap_bids:
        if fmap["suffix"] == "epi":
            fieldmaps["epi"].append((fmap["epi"], layout.get_metadata(fmap["epi"])))

        if fmap["suffix"] == "fieldmap":
            fieldmaps["fieldmap"].append(
                {
                    "magnitude": [
                        (fmap["magnitude"], layout.get_metadata(fmap["magnitude"]))
                    ],
                    "fieldmap": [
                        (fmap["fieldmap"], layout.get_metadata(fmap["fieldmap"]))
                    ],
                }
            )

        if fmap["suffix"] == "phasediff":
            fieldmaps["phasediff"].append(
                {
                    "magnitude": [
                        (fmap[k], layout.get_metadata(fmap[k]))
                        for k in sorted(fmap.keys())
                        if k.startswith("magnitude")
                    ],
                    "phases": [
                        (fmap["phasediff"], layout.get_metadata(fmap["phasediff"]))
                    ],
                }
            )

        if fmap["suffix"] == "phase":
            fieldmaps["phasediff"].append(
                {
                    "magnitude": [
                        (fmap[k], layout.get_metadata(fmap[k]))
                        for k in sorted(fmap.keys())
                        if k.startswith("magnitude")
                    ],
                    "phases": [
                        (fmap[k], layout.get_metadata(fmap[k]))
                        for k in sorted(fmap.keys())
                        if k.startswith("phase")
                    ],
                }
            )

    if fieldmaps and force_syn:
        # syn: True -> Run SyN in addition to fieldmap-based SDC
        fieldmaps["syn"] = True
    elif not fieldmaps and (force_syn or use_syn):
        # syn: False -> Run SyN as only SDC
        fieldmaps["syn"] = False
    return fieldmaps
