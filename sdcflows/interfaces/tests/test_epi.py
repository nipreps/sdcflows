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

import nibabel as nb
import numpy as np

from ..epi import SelectPEVolumes, SortPEBlips


def _write_epi(path, nvols):
    shape = (4, 4, 4, nvols) if nvols > 1 else (4, 4, 4)
    nb.Nifti1Image(np.zeros(shape, dtype='float32'), np.eye(4)).to_filename(path)
    return str(path)


def test_select_pe_volumes_caps(tmp_path, monkeypatch):
    """At most ``max_vols_per_pe`` volumes are kept for each PE direction."""
    monkeypatch.chdir(tmp_path)
    ap = _write_epi(tmp_path / 'ap.nii.gz', 10)
    pa = _write_epi(tmp_path / 'pa.nii.gz', 10)

    result = SelectPEVolumes(
        in_data=[ap, pa],
        pe_dirs_fsl=['y', 'y-'],
        readout_times=[0.05, 0.05],
    ).run()

    assert result.outputs.pe_dirs_fsl == ['y', 'y', 'y', 'y-', 'y-', 'y-']
    assert result.outputs.readout_times == [0.05] * 6
    assert all(nb.load(f).ndim == 3 for f in result.outputs.out_data)


def test_select_pe_volumes_order(tmp_path, monkeypatch):
    """The per-direction budget is filled across runs in the listed order."""
    monkeypatch.chdir(tmp_path)
    ap1 = _write_epi(tmp_path / 'ap1.nii.gz', 2)
    ap2 = _write_epi(tmp_path / 'ap2.nii.gz', 5)
    pa = _write_epi(tmp_path / 'pa.nii.gz', 5)

    result = SelectPEVolumes(
        in_data=[ap1, ap2, pa],
        pe_dirs_fsl=['y', 'y', 'y-'],
        readout_times=[0.05, 0.05, 0.05],
        max_vols_per_pe=3,
    ).run()

    names = [Path(f).name for f in result.outputs.out_data]
    assert result.outputs.pe_dirs_fsl == ['y', 'y', 'y', 'y-', 'y-', 'y-']
    assert names[0].startswith('ap1') and names[1].startswith('ap1')
    assert names[2].startswith('ap2')
    assert names[3].startswith('pa')


def test_sort_pe_blips(tmpdir):
    tmpdir.chdir()

    input_comb = [('x-', 0.08), ('x-', 0.04), ('y-', 0.05), ('y', 0.05), ('x', 0.05)]

    fnames = []
    for i in range(len(input_comb)):
        fnames.append(f'file{i}.nii')
        Path(fnames[-1]).write_text('')

    result = SortPEBlips(
        in_data=fnames,
        pe_dirs_fsl=[pe for pe, _ in input_comb],
        readout_times=[trt for _, trt in input_comb],
    ).run()

    assert result.outputs.out_data == [f'file{i}.nii' for i in (4, 3, 1, 0, 2)]
