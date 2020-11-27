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
