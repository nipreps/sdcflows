# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""Construction tests for the standalone CLI workflow."""

import pytest
from niworkflows.utils.testing import generate_bids_skeleton

from sdcflows import config
from sdcflows.fieldmaps import clear_registry
from sdcflows.workflows.fit.base import init_sdcflows_wf


def _medic_skeleton():
    func = [
        {
            'task': 'rest',
            'echo': echo,
            'part': part,
            'suffix': 'bold',
            'metadata': {
                'EchoTime': te,
                'RepetitionTime': 0.8,
                'TotalReadoutTime': 0.5,
                'PhaseEncodingDirection': 'j',
                'B0FieldIdentifier': 'medic',
                **({'B0FieldSource': 'medic'} if part == 'mag' else {}),
            },
        }
        for echo, te in (('1', 0.0142), ('2', 0.03893))
        for part in ('mag', 'phase')
    ]
    return {'01': [{'anat': [{'suffix': 'T1w', 'metadata': {'EchoTime': 1}}]}, {'func': func}]}


@pytest.mark.parametrize('no_medic', [False, True])
def test_sdcflows_wf_medic(tmp_path, monkeypatch, no_medic):
    """The CLI workflow honors ``--no-medic`` and builds MEDIC without coefficients."""
    from bids.layout import BIDSLayout

    bids_dir = tmp_path / 'bids'
    generate_bids_skeleton(str(bids_dir), _medic_skeleton())

    monkeypatch.setattr(config.execution, 'layout', BIDSLayout(bids_dir, validate=False))
    monkeypatch.setattr(config.execution, 'participant_label', ['01'])
    monkeypatch.setattr(config.execution, 'output_dir', tmp_path / 'out')
    monkeypatch.setattr(config.execution, 'work_dir', tmp_path / 'work')
    monkeypatch.setattr(config.workflow, 'fmapless', False)
    monkeypatch.setattr(config.workflow, 'no_medic', no_medic)

    clear_registry()
    try:
        wf = init_sdcflows_wf()
    finally:
        clear_registry()

    derivs = [n for n in wf.list_node_names() if n.startswith('fmap_derivatives_medic.')]
    if no_medic:
        assert not derivs
    else:
        assert derivs
        assert not any(n.endswith('.ds_coeff') for n in derivs)
