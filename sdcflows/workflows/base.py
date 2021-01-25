# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Estimate fieldmaps for :abbr:`SDC (susceptibility distortion correction)`."""
from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from ..fieldmaps import EstimatorType


LOGGER = logging.getLogger("nipype.workflow")
DEFAULT_MEMORY_MIN_GB = 0.01

INPUT_FIELDS = {
    EstimatorType.MAPPED: ["magnitude", "fieldmap"],
    EstimatorType.PHASEDIFF: ["magnitude", "fieldmap"],
    EstimatorType.PEPOLAR: ["metadata", "in_data"],
    EstimatorType.ANAT: [
        "epi_ref",
        "epi_mask",
        "anat_brain",
        "std2anat_xfm",
        "anat2bold_xfm",
    ],
}


def init_fmap_preproc_wf(
    *,
    estimators,
    omp_nthreads,
    output_dir,
    subject,
    debug=False,
    name="fmap_preproc_wf",
):
    """
    Create and combine estimator workflows.

    Parameters
    ----------
    estimators : :obj:`list` of :py:class:`~sdcflows.fieldmaps.FieldmapEstimator`
        A list of estimators.
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    output_dir : :obj:`str`
        Directory in which to save derivatives
    subject : :obj:`str`
        Participant label for this single-subject workflow.
    debug : :obj:`bool`
        Enable debugging outputs
    name : :obj:`str`, optional
        Workflow name (default: ``"fmap_preproc_wf"``)

    Inputs
    ------
    in_<B0FieldIdentifier>.<field> :
        The workflow generates inputs depending on the estimation strategy.

    Outputs
    -------
    out_<B0FieldIdentifier>.fmap :
        The preprocessed fieldmap.
    out_<B0FieldIdentifier>.fmap_ref :
        The preprocessed fieldmap reference.
    out_<B0FieldIdentifier>.fmap_coeff :
        The preprocessed fieldmap coefficients.

    """
    from .outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

    workflow = Workflow(name=name)

    out_fields = ("fmap", "fmap_ref", "fmap_coeff", "fmap_mask", "fmap_id")
    out_merge = {
        f: pe.Node(niu.Merge(len(estimators)), name=f"out_merge_{f}")
        for f in out_fields
    }
    outputnode = pe.Node(niu.IdentityInterface(fields=out_fields), name="outputnode")

    workflow.connect(
        [
            (mergenode, outputnode, [("out", field)])
            for field, mergenode in out_merge.items()
        ]
    )

    for n, estimator in enumerate(estimators):
        est_wf = estimator.get_workflow(omp_nthreads=omp_nthreads, debug=debug)
        source_files = [str(f.path) for f in estimator.sources]

        out_map = pe.Node(
            niu.IdentityInterface(fields=out_fields), name=f"out_{estimator.bids_id}"
        )
        out_map.inputs.fmap_id = estimator.bids_id

        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(output_dir),
            write_coeff=True,
            bids_fmap_id=estimator.bids_id,
            name=f"fmap_derivatives_wf_{estimator.bids_id}",
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = source_files
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = [
            f.metadata for f in estimator.sources
        ]

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(output_dir),
            fmap_type=str(estimator.method).rpartition(".")[-1].lower(),
            bids_fmap_id=estimator.bids_id,
            name=f"fmap_reports_wf_{estimator.bids_id}",
        )
        fmap_reports_wf.inputs.inputnode.source_files = source_files

        if estimator.method not in (EstimatorType.MAPPED, EstimatorType.PHASEDIFF):
            fields = INPUT_FIELDS[estimator.method]
            inputnode = pe.Node(
                niu.IdentityInterface(fields=fields),
                name=f"in_{estimator.bids_id}",
            )
            # fmt:off
            workflow.connect([
                (inputnode, est_wf, [(f, f"inputnode.{f}") for f in fields])
            ])
            # fmt:on

        # fmt:off
        workflow.connect([
            (est_wf, fmap_derivatives_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
            ]),
            (est_wf, fmap_reports_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_mask", "inputnode.fmap_mask"),
            ]),
            (est_wf, out_map, [
                ("outputnode.fmap", "fmap"),
                ("outputnode.fmap_ref", "fmap_ref"),
                ("outputnode.fmap_coeff", "fmap_coeff"),
                ("outputnode.fmap_mask", "fmap_mask"),
            ]),
        ])
        # fmt:on

        for field, mergenode in out_merge.items():
            workflow.connect(out_map, field, mergenode, f"in{n}")

    return workflow
