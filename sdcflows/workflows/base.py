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
"""Estimate fieldmaps for :abbr:`SDC (susceptibility distortion correction)`."""

from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ..utils.epimanip import get_trt

LOGGER = logging.getLogger("nipype.workflow")
DEFAULT_MEMORY_MIN_GB = 0.01


def init_fmap_preproc_wf(
    *,
    estimators,
    omp_nthreads,
    output_dir,
    subject,
    use_metadata_estimates=False,
    fallback_total_readout_time=None,
    sloppy=False,
    debug=False,
    name="fmap_preproc_wf",
    **kwargs,
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
    sloppy : :obj:`bool`
        Enable faster but less precise calculations
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
    from sdcflows.workflows.fit.pepolar import INPUT_FIELDS as _pepolar_fields
    from sdcflows.workflows.fit.syn import INPUT_FIELDS as _syn_fields
    from sdcflows.workflows.outputs import init_fmap_derivatives_wf, init_fmap_reports_wf
    from sdcflows.fieldmaps import EstimatorType

    INPUT_FIELDS = {
        EstimatorType.ANAT: _syn_fields,
        EstimatorType.PEPOLAR: _pepolar_fields,
    }

    workflow = Workflow(name=name)

    out_fields = ("fmap", "fmap_coeff", "fmap_ref", "fmap_mask", "fmap_id", "method")
    out_merge = {
        f: pe.Node(niu.Merge(len(estimators)), name=f"out_merge_{f}")
        for f in out_fields
    }
    # Fieldmaps and coefficient files can come in pairs, ensure they are not flattened
    out_merge["fmap"].inputs.no_flatten = True
    out_merge["fmap_coeff"].inputs.no_flatten = True

    outputnode = pe.Node(niu.IdentityInterface(fields=out_fields), name="outputnode")

    workflow.connect(
        [
            (mergenode, outputnode, [("out", field)])
            for field, mergenode in out_merge.items()
        ]
    )

    for n, estimator in enumerate(estimators, 1):
        if fallback_total_readout_time is None:
            for f in estimator.sources:
                if f.suffix in ("bold", "dwi", "epi", "sbref", "asl", "m0scan"):
                    try:
                        get_trt(
                            f.metadata,
                            in_file=f.path,
                            use_estimate=use_metadata_estimates,
                        )
                    except ValueError:
                        msg = f"Missing readout timing information for <{f.path}>."
                        raise RuntimeError(msg)

        est_wf = estimator.get_workflow(
            use_metadata_estimates=use_metadata_estimates,
            fallback_total_readout_time=fallback_total_readout_time,
            omp_nthreads=omp_nthreads,
            debug=debug,
            sloppy=sloppy,
            **kwargs,
        )
        source_files = [
            str(f.path) for f in estimator.sources if f.suffix not in ("T1w", "T2w")
        ]

        out_map = pe.Node(
            niu.IdentityInterface(fields=out_fields), name=f"out_{estimator.bids_id}"
        )
        out_map.inputs.fmap_id = estimator.bids_id

        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(output_dir),
            write_coeff=True,
            bids_fmap_id=estimator.bids_id,
            name=f"fmap_derivatives_wf_{estimator.sanitized_id}",
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = source_files
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = [
            f.metadata for f in estimator.sources
        ]

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(output_dir),
            fmap_type=str(estimator.method).rpartition(".")[-1].lower(),
            bids_fmap_id=estimator.bids_id,
            name=f"fmap_reports_wf_{estimator.sanitized_id}",
        )
        fmap_reports_wf.inputs.inputnode.source_files = source_files

        if estimator.method not in (EstimatorType.MAPPED, EstimatorType.PHASEDIFF):
            fields = INPUT_FIELDS[estimator.method]
            inputnode = pe.Node(
                niu.IdentityInterface(fields=fields),
                name=f"in_{estimator.sanitized_id}",
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
                ("outputnode.method", "method")
            ]),
        ])
        # fmt:on

        for field, mergenode in out_merge.items():
            workflow.connect(out_map, field, mergenode, f"in{n}")

    return workflow
