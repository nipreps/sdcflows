# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
"""Test EPI interfaces."""
from pathlib import Path
from ..epi import SortPEBlips


def test_sort_pe_blips(tmpdir):

    tmpdir.chdir()

    input_comb = [("x-", 0.08), ("x-", 0.04), ("y-", 0.05), ("y", 0.05), ("x", 0.05)]

    fnames = []
    for i in range(len(input_comb)):
        fnames.append(f"file{i}.nii")
        Path(fnames[-1]).write_text("")

    result = SortPEBlips(
        in_data=fnames,
        pe_dirs_fsl=[pe for pe, _ in input_comb],
        readout_times=[trt for _, trt in input_comb],
    ).run()

    assert result.outputs.out_data == [f"file{i}.nii" for i in (4, 3, 1, 0, 2)]
