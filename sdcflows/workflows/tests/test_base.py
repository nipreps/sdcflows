"""Test the base workflow."""
import pytest
from ... import fieldmaps as fm
from ...utils.wrangler import find_estimators
from ..base import init_fmap_preproc_wf


@pytest.mark.parametrize(
    "dataset,subject", [("ds000054", "100185"), ("HCP101006", "101006")]
)
def test_fmap_wf(tmpdir, workdir, outdir, bids_layouts, dataset, subject):
    """Test the encompassing of the wrangler and the workflow creator."""
    fm._estimators.clear()
    estimators = find_estimators(bids_layouts[dataset], subject=subject)
    wf = init_fmap_preproc_wf(
        estimators=estimators,
        omp_nthreads=2,
        output_dir=str(outdir),
        subject=subject,
        debug=True,
    )

    for estimator in estimators:
        if estimator.method != fm.EstimatorType.PEPOLAR:
            continue

        inputnode = wf.get_node(f"in_{estimator.bids_id}")
        inputnode.inputs.in_data = [str(f.path) for f in estimator.sources]
        inputnode.inputs.metadata = [f.metadata for f in estimator.sources]

    if workdir:
        wf.base_dir = str(workdir)

    wf.run(plugin="Linear")
