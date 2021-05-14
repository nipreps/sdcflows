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
"""Test the base workflow."""
from pathlib import Path
import os
import pytest
from ... import fieldmaps as fm
from ...utils.wrangler import find_estimators
from ..base import init_fmap_preproc_wf


@pytest.mark.parametrize(
    "dataset,subject", [("ds000054", "100185"), ("HCP101006", "101006")]
)
def test_fmap_wf(tmpdir, workdir, outdir, bids_layouts, dataset, subject):
    """Test the encompassing of the wrangler and the workflow creator."""
    if outdir is None:
        outdir = Path(str(tmpdir))

    outdir = outdir / "test_base" / dataset
    fm._estimators.clear()
    estimators = find_estimators(layout=bids_layouts[dataset], subject=subject)
    wf = init_fmap_preproc_wf(
        estimators=estimators,
        omp_nthreads=2,
        output_dir=str(outdir),
        subject=subject,
        debug=True,
    )

    # PEPOLAR and fieldmap-less solutions typically cannot work directly on the
    # raw inputs. Normally, some ad-hoc massaging and pre-processing is required.
    # For that reason, the inputs cannot be set implicitly by init_fmap_preproc_wf.
    for estimator in estimators:
        if estimator.method != fm.EstimatorType.PEPOLAR:
            continue

        inputnode = wf.get_node(f"in_{estimator.bids_id}")
        inputnode.inputs.in_data = [str(f.path) for f in estimator.sources]
        inputnode.inputs.metadata = [f.metadata for f in estimator.sources]

    if workdir:
        wf.base_dir = str(workdir)

    if os.getenv("GITHUB_ACTIONS") != "true":
        wf.run(plugin="Linear")
