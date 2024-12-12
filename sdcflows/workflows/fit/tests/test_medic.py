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
"""Test phase-difference type of fieldmaps."""
from pathlib import Path
from json import loads

import pytest

from ..medic import init_medic_wf, Workflow


@pytest.mark.slow
def test_medic(tmpdir, datadir, workdir, outdir):
    """Test creation of the workflow."""
    tmpdir.chdir()

    pattern = 'ds005250/sub-04/ses-2/func/*_part-mag_bold.nii.gz'
    magnitude_files = sorted(datadir.glob(pattern))
    phase_files = [f.with_name(f.name.replace("part-mag", "part-phase")) for f in magnitude_files]
    metadata_dicts = [
        loads(Path(f.with_name(f.name.replace('.nii.gz', '.json'))).read_text())
        for f in magnitude_files
    ]

    wf = Workflow(name=f"medic_{magnitude_files[0].name.replace('.nii.gz', '').replace('-', '_')}")
    medic_wf = init_medic_wf()
    medic_wf.inputs.inputnode.magnitude = magnitude_files
    medic_wf.inputs.inputnode.phase = phase_files
    medic_wf.inputs.inputnode.metadata = metadata_dicts

    if outdir:
        from ...outputs import init_fmap_derivatives_wf, init_fmap_reports_wf

        outdir = outdir / "unittests" / magnitude_files[0].split("/")[0]
        fmap_derivatives_wf = init_fmap_derivatives_wf(
            output_dir=str(outdir),
            write_coeff=True,
            bids_fmap_id="phasediff_id",
        )
        fmap_derivatives_wf.inputs.inputnode.source_files = [str(f) for f in magnitude_files]
        fmap_derivatives_wf.inputs.inputnode.fmap_meta = metadata_dicts

        fmap_reports_wf = init_fmap_reports_wf(
            output_dir=str(outdir),
            fmap_type='medic',
        )
        fmap_reports_wf.inputs.inputnode.source_files = [str(f) for f in magnitude_files]

        wf.connect([
            (medic_wf, fmap_reports_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
            ]),
            (medic_wf, fmap_derivatives_wf, [
                ("outputnode.fmap", "inputnode.fieldmap"),
            ]),
        ])  # fmt:skip
    else:
        wf.add_nodes([medic_wf])

    if workdir:
        wf.base_dir = str(workdir)

    wf.run(plugin="Linear")
