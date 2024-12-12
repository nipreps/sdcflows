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
"""Processing of dynamic field maps from complex-valued multi-echo BOLD data."""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from sdcflows.interfaces.fmap import MEDIC, PhaseMap2rads2

INPUT_FIELDS = ("magnitude", "phase", "metadata")


def init_medic_wf(name="medic_wf"):
    """Create the MEDIC dynamic field estimation workflow.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.medic import init_medic_wf

            wf = init_medic_wf()   # doctest: +SKIP

    Parameters
    ----------
    name : :obj:`str`
        Name for this workflow

    Inputs
    ------
    magnitude : :obj:`list` of :obj:`str`
        A list of echo-wise magnitude EPI files that will be fed into MEDIC.
    phase : :obj:`list` of :obj:`str`
        A list of echo-wise phase EPI files that will be fed into MEDIC.
    metadata : :obj:`list` of :obj:`dict`
        List of metadata dictionaries corresponding to each of the input magnitude files.

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap time series file. Units are Hertz.
    displacement : :obj:`list` of :obj:`str`
        Path to the displacement time series files. Units are mm.
    method: :obj:`str`
        Short description of the estimation method that was run.

    Notes
    -----
    This workflow performs minimal preparation before running the MEDIC algorithm,
    as implemented in ``vandandrew/warpkit``.

    Any downstream processing piplines that use this workflow should include
    the following references in their boilerplate BibTeX file:

    - medic: https://doi.org/10.1101/2023.11.28.568744
    """
    workflow = Workflow(name=name)

    workflow.__desc__ = """\
Volume-wise *B<sub>0</sub>* nonuniformity maps (or *fieldmaps*) were estimated from
complex-valued, multi-echo EPI data using the MEDIC algorithm (@medic).
"""

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fmap", "displacement", "method"]),
        name="outputnode",
    )
    outputnode.inputs.method = "MEDIC"

    # Write metadata dictionaries to JSON files
    write_metadata = pe.MapNode(
        niu.Function(
            input_names=["metadata"],
            output_names=["out_file"],
            function=write_json,
        ),
        iterfield=["metadata"],
        name="write_metadata",
    )
    workflow.connect([(inputnode, write_metadata, [("metadata", "metadata")])])

    # Convert phase to radians (-pi to pi, not 0 to 2pi)
    phase2rad = pe.MapNode(
        PhaseMap2rads2(),
        iterfield=["in_file"],
        name="phase2rad",
    )
    workflow.connect([(inputnode, phase2rad, [("phase", "in_file")])])

    medic = pe.Node(
        MEDIC(),
        name="medic",
    )
    workflow.connect([
        (inputnode, medic, [("magnitude", "mag_files")]),
        (write_metadata, medic, [("out_file", "metadata")]),
        (phase2rad, medic, [("out_file", "phase_files")]),
        (medic, outputnode, [
            ("native_field_map", "fmap"),
            ("displacement_map", "displacement"),
        ]),
    ])  # fmt:skip

    return workflow


def write_json(metadata):
    """Write a dictionary to a JSON file."""
    import json
    import os

    out_file = os.path.abspath("metadata.json")
    with open(out_file, "w") as fobj:
        json.dump(metadata, fobj, sort_keys=True, indent=4)

    return out_file
