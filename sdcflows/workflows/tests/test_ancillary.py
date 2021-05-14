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
"""Check the tools submodule."""
import os
import pytest
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.interfaces.reportlets.masks import SimpleShowMaskRPT
from ..ancillary import init_brainextraction_wf


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true", reason="this is GH Actions")
@pytest.mark.parametrize("folder", ["magnitude/ds000054", "magnitude/ds000217"])
def test_brainmasker(tmpdir, datadir, workdir, outdir, folder):
    """Exercise the brain masking tool."""
    tmpdir.chdir()

    wf = pe.Workflow(name=f"test_mask_{folder.replace('/', '_')}")
    if workdir:
        wf.base_dir = str(workdir)

    input_files = [
        str(f) for f in (datadir / "brain-extraction-tests" / folder).glob("*.nii.gz")
    ]

    inputnode = pe.Node(niu.IdentityInterface(fields=("in_file",)), name="inputnode")
    inputnode.iterables = ("in_file", input_files)
    merger = pe.Node(niu.Function(function=_merge), name="merger")

    brainmask_wf = init_brainextraction_wf()

    # fmt:off
    wf.connect([
        (inputnode, merger, [("in_file", "in_file")]),
        (merger, brainmask_wf, [("out", "inputnode.in_file")]),
    ])
    # fmt:on

    if outdir:
        out_path = outdir / "masks" / folder.split("/")[-1]
        out_path.mkdir(exist_ok=True, parents=True)
        report = pe.Node(SimpleShowMaskRPT(), name="report")
        report.interface._always_run = True

        def _report_name(fname, out_path):
            from pathlib import Path

            return str(
                out_path
                / Path(fname)
                .name.replace(".nii", "_mask.svg")
                .replace("_magnitude", "_desc-magnitude")
                .replace(".gz", "")
            )

        # fmt: off
        wf.connect([
            (inputnode, report, [(("in_file", _report_name, out_path), "out_report")]),
            (brainmask_wf, report, [("outputnode.out_mask", "mask_file"),
                                    ("outputnode.out_file", "background_file")]),
        ])
        # fmt: on

    wf.run()


def _merge(in_file):
    import nibabel as nb
    import numpy as np

    img = nb.squeeze_image(nb.load(in_file))

    data = np.asanyarray(img.dataobj)
    if data.ndim == 3:
        return in_file

    from pathlib import Path

    data = data.mean(-1)
    out_file = (Path() / "merged.nii.gz").absolute()
    img.__class__(data, img.affine, img.header).to_filename(out_file)
    return str(out_file)
