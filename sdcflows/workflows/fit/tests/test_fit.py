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
"""Test that workflows build."""
import pytest
import sys

from .. import fieldmap  # noqa
from .. import pepolar  # noqa
from .. import syn  # noqa


@pytest.mark.parametrize(
    "workflow,kwargs",
    (
        ("sdcflows.workflows.fit.fieldmap.init_fmap_wf", {"mode": "mapped"}),
        ("sdcflows.workflows.fit.fieldmap.init_fmap_wf", {}),
        ("sdcflows.workflows.fit.fieldmap.init_phdiff_wf", {"omp_nthreads": 1}),
        ("sdcflows.workflows.fit.pepolar.init_3dQwarp_wf", {}),
        ("sdcflows.workflows.fit.pepolar.init_topup_wf", {}),
        ("sdcflows.workflows.fit.syn.init_syn_sdc_wf", {"omp_nthreads": 1}),
    ),
)
def test_build_1(workflow, kwargs):
    """Make sure the workflow builds."""
    module = ".".join(workflow.split(".")[:-1])
    func = workflow.split(".")[-1]
    getattr(sys.modules[module], func)(**kwargs)
