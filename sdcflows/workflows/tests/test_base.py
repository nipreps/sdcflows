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

import pytest

from sdcflows import fieldmaps as fm
from sdcflows.utils.wrangler import find_estimators
from sdcflows.workflows.base import init_fmap_preproc_wf


@pytest.mark.veryslow
@pytest.mark.slow
@pytest.mark.parametrize('dataset,subject', [('ds000054', '100185'), ('HCP101006', '101006')])
def test_fmap_wf(tmpdir, workdir, outdir, bids_layouts, dataset, subject):
    """Test the encompassing of the wrangler and the workflow creator."""
    if outdir is None:
        outdir = Path(str(tmpdir))

    outdir = outdir / 'test_base' / dataset
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

        inputnode = wf.get_node(f'in_{estimator.sanitized_id}')
        inputnode.inputs.in_data = [str(f.path) for f in estimator.sources]
        inputnode.inputs.metadata = [f.metadata for f in estimator.sources]

    if workdir:
        wf.base_dir = str(workdir)

    res = wf.run(plugin='Linear')

    # Regression test for when out_merge_fmap_coeff was flattened and would
    # have twice as many elements as the other nodes
    assert all(
        len(node.result.outputs.out) == len(estimators)
        for node in res.nodes
        if node.name.startswith('out_merge_')
    )


def test_fmap_preproc_wf_medic(tmp_path, dsA_dir):
    """MEDIC's raw inputs survive the outer ``in_<id>`` node, and no coeffs are sunk."""
    import shutil

    src = dsA_dir / 'sub-01' / 'func' / 'sub-01_task-rest_bold.nii.gz'
    files = []
    for echo in (1, 2):
        for part in ('mag', 'phase'):
            dst = tmp_path / f'sub-01_task-rest_echo-{echo}_part-{part}_bold.nii.gz'
            shutil.copy(src, dst)
            metadata = {
                'EchoTime': 0.01 * echo,
                'TotalReadoutTime': 0.05,
                'PhaseEncodingDirection': 'j',
            }
            files.append(fm.FieldmapFile(dst, metadata=metadata))

    fm.clear_registry()
    try:
        estimator = fm.FieldmapEstimation(files)
        wf = init_fmap_preproc_wf(
            estimators=[estimator],
            omp_nthreads=1,
            output_dir=str(tmp_path / 'out'),
            subject='01',
        )
    finally:
        fm.clear_registry()

    est_inputs = estimator.get_workflow().inputs.inputnode
    inputnode = wf.get_node(f'in_{estimator.sanitized_id}')
    assert inputnode.inputs.phase == est_inputs.phase
    assert inputnode.inputs.magnitude == est_inputs.magnitude
    assert inputnode.inputs.metadata == est_inputs.metadata

    derivs = wf.get_node(f'fmap_derivatives_wf_{estimator.sanitized_id}')
    assert derivs.get_node('ds_coeff') is None
