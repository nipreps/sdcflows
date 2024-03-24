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
from nipype.interfaces.fsl import Split as FSLSplit
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from sdcflows.interfaces.fmap import PhaseMap2rads, ROMEO
from sdcflows.utils.misc import calculate_diffs, medic_automask

INPUT_FIELDS = ("magnitude", "phase", "metadata")


def init_medic_wf(
    n_volumes,
    echo_times,
    automask,
    automask_dilation,
    omp_nthreads=1,
    sloppy=False,
    debug=False,
    name="medic_wf",
):
    """
    Create the PEPOLAR field estimation workflow based on FSL's ``topup``.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.pepolar import init_topup_wf
            wf = init_topup_wf()

    Parameters
    ----------
    sloppy : :obj:`bool`
        Whether a fast configuration of topup (less accurate) should be applied.
    debug : :obj:`bool`
        Run in debug mode
    name : :obj:`str`
        Name for this workflow
    omp_nthreads : :obj:`int`
        Parallelize internal tasks across the number of CPUs given by this option.

    Inputs
    ------
    magnitude : :obj:`list` of :obj:`str`
        A list of echo-wise magnitude EPI files that will be fed into MEDIC.
    phase : :obj:`list` of :obj:`str`
        A list of echo-wise phase EPI files that will be fed into MEDIC.
    metadata : :obj:`list` of :obj:`dict`
        A list of dictionaries containing the metadata corresponding to each file
        in ``in_data``.

    Outputs
    -------
    fmap : :obj:`str`
        The path of the estimated fieldmap.
    fmap_ref : :obj:`str`
        The path of an unwarped conversion of files in ``in_data``.
    fmap_mask : :obj:`str`
        The path of mask corresponding to the ``fmap_ref`` output.
    fmap_coeff : :obj:`str` or :obj:`list` of :obj:`str`
        The path(s) of the B-Spline coefficients supporting the fieldmap.
    method: :obj:`str`
        Short description of the estimation method that was run.

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "fmap",
                "fmap_ref",
                "fmap_coeff",
                "fmap_mask",
                "jacobians",
                "xfms",
                "out_warps",
                "method",
            ],
        ),
        name="outputnode",
    )
    outputnode.inputs.method = "MEDIC"

    n_echoes = len(echo_times)

    # Convert phase to radians (-pi to pi)
    phase2rad = pe.MapNode(
        PhaseMap2rads(),
        iterfield=["in_file"],
        name="phase2rad",
    )
    workflow.connect([(inputnode, phase2rad, [("phase", "in_file")])])

    # Split phase data into single-frame, multi-echo 4D files
    concat_phase_across_echoes = pe.MapNode(
        niu.Merge(numinputs=n_echoes),
        iterfield=[f"in{i + 1}" for i in range(n_echoes)],
        name="concat_phase_across_echoes",
    )
    concat_mag_across_echoes = pe.MapNode(
        niu.Merge(numinputs=n_echoes),
        iterfield=[f"in{i + 1}" for i in range(n_echoes)],
        name="concat_mag_across_echoes",
    )
    for i_echo in range(n_echoes):
        select_phase_echo = pe.Node(
            niu.Select(index=i_echo),
            name=f"select_phase_echo_{i_echo:02d}",
        )
        workflow.connect([(phase2rad, select_phase_echo, [("out_file", "inlist")])])

        split_phase = pe.Node(
            FSLSplit(dimension="t"),
            name=f"split_phase_{i_echo:02d}",
        )
        workflow.connect([
            (select_phase_echo, split_phase, [("out", "in_file")]),
            (split_phase, concat_phase_across_echoes, [("out_files", f"in{i_echo + 1}")]),
        ])  # fmt:skip

        # Split magnitude data into single-frame, multi-echo 4D files
        select_mag_echo = pe.Node(
            niu.Select(index=i_echo),
            name=f"select_mag_echo_{i_echo:02d}",
        )
        workflow.connect([(inputnode, select_mag_echo, [("magnitude", "inlist")])])

        split_mag = pe.Node(
            FSLSplit(dimension="t"),
            name=f"split_mag_{i_echo:02d}",
        )
        workflow.connect([
            (select_mag_echo, split_mag, [("out", "in_file")]),
            (split_mag, concat_mag_across_echoes, [("out_files", f"in{i_echo + 1}")]),
        ])  # fmt:skip

    for volume in range(n_volumes):
        process_volume_wf = init_process_volume_wf(
            echo_times,
            automask,
            automask_dilation,
            name=f"process_volume_{volume:02d}_wf",
        )

    mask_buffer = pe.Node(
        niu.IdentityInterface(fields=["mask_file"]),
        name="mask_buffer",
    )
    if automask:
        # the theory goes like this, the magnitude/otsu base mask can be too aggressive
        # occasionally and the voxel quality mask can get extra voxels that are not brain,
        # but is noisy so we combine the two masks to get a better mask

        # Use ROMEO's voxel-quality command
        voxqual = pe.MapNode(
            ROMEO(write_quality=True, echo_times=echo_times),
            name="voxqual",
            iterfield=["in_file"],
        )
        workflow.connect([
            (concat_mag_across_echoes, voxqual, [("out", "mag_file")]),
            (concat_phase_across_echoes, voxqual, [("out", "phase_file")]),
        ])  # fmt:skip

        # Then use skimage's otsu thresholding to get a mask
        # and do a bunch of other stuff
        automask_medic = pe.Node(
            niu.Function(
                input_names=["mag_file", "voxel_quality", "echo_times", "automask_dilation"],
                output_names=["mask_file"],
                function=medic_automask,
            ),
            name="automask_medic",
        )
        automask_medic.inputs.echo_times = echo_times
        automask_medic.inputs.automask_dilation = automask_dilation
        workflow.connect([
            (concat_mag_across_echoes, automask_medic, [("out", "mag_file")]),
            (voxqual, automask_medic, [("quality_file", "voxel_quality")]),
            (automask_medic, mask_buffer, [("out_file", "mask_file")]),
        ])  # fmt:skip
    else:
        mask_buffer.inputs.mask_file = "NULL"

    # Do MCPC-3D-S algo to compute phase offset
    create_diffs = pe.MapNode(
        niu.Function(
            input_names=["mag_file", "phase_file"],
            output_names=["diff_file"],
            function=calculate_diffs,
        ),
        iterfield=["mag_file", "phase_file"],
        name="create_diffs",
    )
    workflow.connect([
        (concat_mag_across_echoes, create_diffs, [("out", "mag_file")]),
        (concat_phase_across_echoes, create_diffs, [("out", "phase_file")]),
    ])  # fmt:skip

    # Unwrap phase data with ROMEO
    unwrap_phase = pe.MapNode(
        ROMEO(
            echo_times=echo_times,
            weights="romeo",
            correct_global=True,
            maxseeds=1,
        ),
        name="unwrap_phase",
        iterfield=["phase_file"],
    )
    workflow.connect([(concat_phase_across_echoes, unwrap_phase, [("out", "phase_file")])])

    # Re-split the unwrapped phase data

    # Re-combine into echo-wise time series

    # Check temporal consistency of phase unwrapping

    # Compute field maps

    # Apply SVD filter to field maps

    return workflow
