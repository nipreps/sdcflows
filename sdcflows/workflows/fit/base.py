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
"""Build a dataset-wide estimation workflow."""


def init_sdcflows_wf():
    """Create a multi-subject, multi-estimator *SDCFlows* workflow."""
    from nipype.pipeline.engine import Workflow
    from niworkflows.utils.bids import collect_participants

    from sdcflows import config
    from sdcflows.utils.wrangler import find_estimators
    from sdcflows.workflows.outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

    # Create parent workflow
    workflow = Workflow(name="sdcflows_wf")
    workflow.base_dir = config.execution.work_dir

    subjects = collect_participants(
        config.execution.layout,
        config.execution.participant_label,
    )
    estimators_record = {}
    for subject in subjects:
        estimators_record[subject] = find_estimators(
            layout=config.execution.layout,
            subject=subject,
            fmapless=config.workflow.fmapless,
            logger=config.loggers.cli,
        )

    for subject, sub_estimators in estimators_record.items():
        for estim in sub_estimators:
            estim_wf = estim.get_workflow(
                omp_nthreads=config.nipype.omp_nthreads,
                sloppy=False,
                debug=False,
            )

            derivs_wf = init_fmap_derivatives_wf(
                output_dir=config.execution.output_dir,
                bids_fmap_id=estim.bids_id,
                write_coeff=True,
                name=f"fmap_derivatives_{estim.sanitized_id}",
            )

            source_paths = [
                str(source.path.absolute()) for source in estim.sources
            ]
            derivs_wf.inputs.inputnode.source_files = source_paths
            derivs_wf.inputs.inputnode.fmap_meta = [
                source.metadata for source in estim.sources
            ]

            reportlets_wf = init_fmap_reports_wf(
                fmap_type=estim.method,
                output_dir=config.execution.output_dir,
                bids_fmap_id=estim.bids_id,
                name=f"fmap_reports_{estim.sanitized_id}",
            )
            reportlets_wf.inputs.inputnode.source_files = source_paths

            # fmt:off
            workflow.connect([
                (estim_wf, derivs_wf, [
                    ("outputnode.fmap", "inputnode.fieldmap"),
                    ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                    ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
                ]),
                (estim_wf, reportlets_wf, [
                    ("outputnode.fmap", "inputnode.fieldmap"),
                    ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                    ("outputnode.fmap_mask", "inputnode.fmap_mask"),
                ]),

            ])
            # fmt:on

    return workflow
