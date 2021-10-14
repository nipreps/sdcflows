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
"""Test pepolar type of fieldmaps."""
import os
import pytest
from nipype.pipeline import engine as pe

from ..pepolar import init_topup_wf


@pytest.mark.skipif(os.getenv("TRAVIS") == "true", reason="this is TravisCI")
@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="this is GH Actions")
@pytest.mark.parametrize("ds", ("ds001771", "HCP101006"))
def test_topup_wf(tmpdir, bids_layouts, workdir, outdir, ds):
    """Test preparation workflow."""
    layout = bids_layouts[ds]
    epi_path = sorted(
        layout.get(suffix="epi", extension=["nii", "nii.gz"], scope="raw"),
        key=lambda k: k.path,
    )
    in_data = [f.path for f in epi_path]

    wf = pe.Workflow(name=f"topup_{ds}")
    topup_wf = init_topup_wf(omp_nthreads=2, debug=True, sloppy=True)
    metadata = [layout.get_metadata(f.path) for f in epi_path]

    topup_wf.inputs.inputnode.in_data = in_data
    topup_wf.inputs.inputnode.metadata = metadata

    if outdir:
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        outdir = outdir / "unittests" / f"topup_{ds}"
        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(outdir),
            write_coeff=True,
            bids_fmap_id="pepolar_id",
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = in_data
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = metadata

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(outdir),
            fmap_type="pepolar",
        )
        fmap_reports_wf.inputs.inputnode.source_files = in_data

        # fmt: off
        wf.connect([
            (topup_wf, fmap_reports_wf, [("outputnode.fmap", "inputnode.fieldmap"),
                                         ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                                         ("outputnode.fmap_mask", "inputnode.fmap_mask")]),
            (topup_wf, fmap_derivatives_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
                ("outputnode.fmap_ref", "inputnode.fmap_ref"),
                ("outputnode.fmap_coeff", "inputnode.fmap_coeff"),
            ]),
        ])
        # fmt: on
    else:
        wf.add_nodes([topup_wf])

    if workdir:
        wf.base_dir = str(workdir)

    wf.run(plugin="Linear")
