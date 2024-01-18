# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
"""Check the CLI."""
from importlib import reload
import pytest
from niworkflows.utils.testing import generate_bids_skeleton

from sdcflows.cli.main import main as cli_finder_wrapper
from sdcflows.fieldmaps import clear_registry

OUTPUT = """\
Estimation for <{path}> complete. Found:
\tsub-01
\t\tFieldmapEstimation(sources=<2 files>, method=<EstimatorType.PEPOLAR: 2>, \
bids_id='{estimator_id}')
\t\t\tj-\tfmap/sub-01_dir-AP_epi.nii.gz
\t\t\tj\tfmap/sub-01_dir-PA_epi.nii.gz
\tsub-02
\t\tNo estimators found
"""

intendedfor_config = {
    "01": [
        {"anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}]},
        {
            "fmap": [
                {
                    "dir": "AP",
                    "suffix": "epi",
                    "metadata": {
                        "TotalReadoutTime": 0.1,
                        "PhaseEncodingDirection": "j-",
                        "IntendedFor": "func/sub-01_task-rest_bold.nii.gz",
                    },
                },
                {
                    "dir": "PA",
                    "suffix": "epi",
                    "metadata": {
                        "TotalReadoutTime": 0.1,
                        "PhaseEncodingDirection": "j",
                        "IntendedFor": "func/sub-01_task-rest_bold.nii.gz",
                    },
                },
            ]
        },
        {
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "RepetitionTime": 0.8,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ]
        },
    ],
    "02": [{"anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}]}],
}


b0field_config = {
    "01": [
        {"anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}]},
        {
            "fmap": [
                {
                    "dir": "AP",
                    "suffix": "epi",
                    "metadata": {
                        "B0FieldIdentifier": "pepolar",
                        "TotalReadoutTime": 0.1,
                        "PhaseEncodingDirection": "j-",
                    },
                },
                {
                    "dir": "PA",
                    "suffix": "epi",
                    "metadata": {
                        "B0FieldIdentifier": "pepolar",
                        "TotalReadoutTime": 0.1,
                        "PhaseEncodingDirection": "j",
                    },
                },
            ]
        },
        {
            "func": [
                {
                    "task": "rest",
                    "suffix": "bold",
                    "metadata": {
                        "B0FieldSource": "pepolar",
                        "RepetitionTime": 0.8,
                        "PhaseEncodingDirection": "j",
                    },
                }
            ]
        },
    ],
    "02": [{"anat": [{"suffix": "T1w", "metadata": {"EchoTime": 1}}]}],
}


@pytest.mark.parametrize(
    "test_id,config,estimator_id",
    [
        ("intendedfor", intendedfor_config, "auto_00000"),
        ("b0field", b0field_config, "pepolar"),
    ],
)
def test_cli_finder_wrapper(tmp_path, capsys, test_id, config, estimator_id):
    """Test the CLI with --dry-run."""
    import sdcflows.config as sc

    # Reload is necessary to clean-up the layout config between parameterized runs
    reload(sc)

    path = (tmp_path / test_id).absolute()
    generate_bids_skeleton(path, config)
    with pytest.raises(SystemExit) as wrapped_exit:
        cli_finder_wrapper([str(path), str(tmp_path / "out"), "participant", "--dry-run"])

    assert wrapped_exit.value.code == 0
    output = OUTPUT.format(path=path, estimator_id=estimator_id)
    out, _ = capsys.readouterr()
    assert out == output
    clear_registry()
