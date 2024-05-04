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

INPUT_FIELDS = ("magnitude", "phase")


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
    metadata : :obj:`list` of :obj:`str`
        List of JSON files corresponding to each of the input magnitude files.

    Outputs
    -------
    fieldmap : :obj:`str`
        The path of the estimated fieldmap time series file. Units are Hertz.
    displacement : :obj:`list` of :obj:`str`
        Path to the displacement time series files. Units are mm.
    method: :obj:`str`
        Short description of the estimation method that was run.

    Notes
    -----
    This is a translation of the MEDIC algorithm, as implemented in ``vandandrew/warpkit``
    (specifically the function ``unwrap_and_compute_field_maps``), into a Nipype workflow.

    """
    workflow = Workflow(name=name)

    workflow.__desc__ = """\
A dynamic fieldmap was estimated from multi-echo EPI data using the MEDIC algorithm (@medic).
"""

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fieldmap", "displacement", "method"]),
        name="outputnode",
    )
    outputnode.inputs.method = "MEDIC"

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
        (inputnode, medic, [
            ("magnitude", "magnitude"),
            ("metadata", "metadata"),
        ]),
        (phase2rad, medic, [("out_file", "phase")]),
        (medic, outputnode, [
            ("native_field_map", "fieldmap"),
            ("displacement_map", "displacement"),
        ]),
    ])  # fmt:skip

    return workflow


def init_process_volume_wf(
    echo_times,
    automask=True,
    automask_dilation=3,
    name="process_volume_wf",
):
    """Process a single volume of multi-echo data according to the MEDIC method.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.medic import init_process_volume_wf

            wf = init_process_volume_wf(
                echo_times=[0.015, 0.030, 0.045, 0.06],
                automask=True,
                automask_dilation=3,
            )   # doctest: +SKIP

    Parameters
    ----------
    echo_times : :obj:`list` of :obj:`float`
        The echo times of the multi-echo data.
    automask : :obj:`bool`
        Whether to automatically generate a mask for the fieldmap.
    automask_dilation : :obj:`int`
        The number of voxels by which to dilate the automatically generated mask.
    name : :obj:`str`
        The name of the workflow.

    Inputs
    ------
    magnitude : :obj:`str`
        One volume of magnitude EPI data, concatenated across echoes.
    phase : :obj:`str`
        One volume of phase EPI data, concatenated across echoes.
        Must already be scaled from -pi to pi.
    mask : :obj:`str`
        The brain mask that will be used to constrain the fieldmap estimation.
        If ``automask`` is True, this mask will be modified.
        Otherwise, it will be returned in the outputnode unmodified.

    Outputs
    -------
    phase_unwrapped : :obj:`str`
        Unwrapped phase in radians.
    mask : :obj:`str`
        Path to a brain mask that can be used to constrain the fieldmap estimation.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["magnitude", "phase", "mask"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["phase_unwrapped", "mask"]),
        name="outputnode",
    )

    mask_buffer = pe.Node(
        niu.IdentityInterface(fields=["mask"]),
        name="mask_buffer",
    )
    if automask:
        # the theory goes like this, the magnitude/otsu base mask can be too aggressive
        # occasionally and the voxel quality mask can get extra voxels that are not brain,
        # but is noisy so we combine the two masks to get a better mask

        # Use ROMEO's voxel-quality command
        # XXX: In warpkit Andrew creates a "mag" image of all ones.
        # With the current version (at least on some test data),
        # there are NaNs in the voxel quality map.
        voxqual = pe.Node(
            ROMEO(write_quality=True, echo_times=echo_times, mask="robustmask"),
            name="voxqual",
        )
        workflow.connect([
            (inputnode, voxqual, [
                ("magnitude", "mag_file"),
                ("phase", "phase_file"),
            ]),
        ])  # fmt:skip

        # Then use skimage's otsu thresholding to get a mask and do a bunch of other stuff
        automask_medic = pe.Node(
            niu.Function(
                input_names=["mag_file", "voxel_quality", "echo_times", "automask_dilation"],
                output_names=["mask", "masksum_file"],
                function=medic_automask,
            ),
            name="automask_medic",
        )
        automask_medic.inputs.echo_times = echo_times
        automask_medic.inputs.automask_dilation = automask_dilation
        workflow.connect([
            (inputnode, automask_medic, [("magnitude", "mag_file")]),
            (voxqual, automask_medic, [("quality_file", "voxel_quality")]),
            (automask_medic, mask_buffer, [("mask", "mask")]),
        ])  # fmt:skip
    else:
        workflow.connect([(inputnode, mask_buffer, [("mask", "mask")])])

    workflow.connect([(mask_buffer, outputnode, [("mask", "mask")])])

    # Do MCPC-3D-S algo to compute phase offset
    mcpc_3d_s_wf = init_mcpc_3d_s_wf(wrap_limit=False, name="mcpc_3d_s_wf")
    mcpc_3d_s_wf.inputs.inputnode.echo_times = echo_times
    workflow.connect([
        (inputnode, mcpc_3d_s_wf, [
            ("magnitude", "inputnode.magnitude"),
            ("phase", "inputnode.phase"),
        ]),
        (mask_buffer, mcpc_3d_s_wf, [("mask", "inputnode.mask")]),
    ])  # fmt:skip

    # remove offset from phase data
    remove_offset = pe.Node(
        niu.Function(
            input_names=["phase", "offset"],
            output_names=["phase_modified"],
            function=subtract_offset,
        ),
        name="remove_offset",
    )
    workflow.connect([
        (inputnode, remove_offset, [("phase", "phase")]),
        (mcpc_3d_s_wf, remove_offset, [("outputnode.offset", "offset")]),
    ])  # fmt:skip

    # Unwrap the modified phase data with ROMEO
    unwrap_phase = pe.Node(
        ROMEO(
            echo_times=echo_times,
            weights="romeo",
            correct_global=True,
            max_seeds=1,
            merge_regions=False,
            correct_regions=False,
        ),
        name="unwrap_phase",
    )
    workflow.connect([
        (inputnode, unwrap_phase, [("magnitude", "mag_file")]),
        (mask_buffer, unwrap_phase, [("mask", "mask")]),
        (remove_offset, unwrap_phase, [("phase_modified", "phase_file")]),
    ])  # fmt:skip

    # Global mode correction
    global_mode_corr = pe.Node(
        niu.Function(
            input_names=["magnitude", "unwrapped", "mask", "echo_times"],
            output_names=["unwrapped"],
            function=global_mode_correction,
        ),
        name="global_mode_corr",
    )
    global_mode_corr.inputs.echo_times = echo_times
    workflow.connect([
        (inputnode, global_mode_corr, [("magnitude", "magnitude")]),
        (unwrap_phase, global_mode_corr, [("out_file", "unwrapped")]),
        (mask_buffer, global_mode_corr, [("mask", "mask")]),
        (global_mode_corr, outputnode, [("unwrapped", "phase_unwrapped")]),
    ])  # fmt:skip

    return workflow


def init_mcpc_3d_s_wf(wrap_limit, name):
    """Estimate and remove phase offset with MCPC-3D-S algorithm.

    Parameters
    ----------
    wrap_limit : bool
        If True, this turns off some heuristics for phase unwrapping.
    name : str
        The name of the workflow.

    Inputs
    ------
    magnitude : str
        The path to the magnitude image. A single volume, concatenated across echoes.
    phase : str
        The path to the phase image. A single volume, concatenated across echoes.
    echo_times : list of float
        The echo times of the multi-echo data.
    mask : str
        The path to the brain mask mask.

    Outputs
    -------
    offset : str
        The path to the estimated phase offset.
    unwrapped_diff : str
        The path to the unwrapped phase difference image.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "magnitude",
                "phase",
                "echo_times",
                "mask",
            ],
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["offset", "unwrapped_diff"]),
        name="outputnode",
    )

    # Calculate magnitude and phase differences from first two echoes
    calc_diffs = pe.Node(
        niu.Function(
            input_names=["magnitude", "phase"],
            output_names=["mag_diff_file", "phase_diff_file"],
            function=calculate_diffs2,
        ),
        name="calc_diffs",
    )
    workflow.connect([
        (inputnode, calc_diffs, [
            ("magnitude", "magnitude"),
            ("phase", "phase"),
        ]),
    ])  # fmt:skip

    # Unwrap difference images
    unwrap_diffs = pe.Node(
        ROMEO(
            weights="romeo",
            correct_global=True,
        ),
        name="unwrap_diffs",
    )
    workflow.connect([
        (inputnode, unwrap_diffs, [("mask", "mask")]),
        (calc_diffs, unwrap_diffs, [
            ("mag_diff_file", "mag_file"),
            ("phase_diff_file", "phase_file"),
        ]),
    ])  # fmt:skip

    # Calculate voxel mask
    create_mask = pe.Node(
        niu.Function(
            input_names=["magnitude", "extra_dilation"],
            output_names=["mask"],
            function=create_brain_mask,
        ),
        name="create_mask",
    )
    create_mask.inputs.extra_dilation = -2  # hardcoded in warpkit
    workflow.connect([(inputnode, create_mask, [("magnitude", "magnitude")])])

    # Calculate initial offset estimate
    calc_offset = pe.Node(
        niu.Function(
            input_names=["phase", "unwrapped_diff", "echo_times"],
            output_names=["offset"],
            function=calculate_offset,
        ),
        name="calc_offset",
    )
    workflow.connect([
        (inputnode, calc_offset, [
            ("phase", "phase"),
            ("echo_times", "echo_times"),
        ]),
        (unwrap_diffs, calc_offset, [("out_file", "unwrapped_diff")]),
    ])  # fmt:skip

    # Get the new phase
    calc_proposed_phase = pe.Node(
        niu.Function(
            input_names=["phase", "offset"],
            output_names=["proposed_phase"],
            function=subtract_offset,
        ),
        name="calc_proposed_phase",
    )
    workflow.connect([
        (inputnode, calc_proposed_phase, [("phase", "phase")]),
        (calc_offset, calc_proposed_phase, [("offset", "offset")]),
    ])  # fmt:skip

    # Compute the dual-echo field map
    dual_echo_wf = init_dual_echo_wf(name="dual_echo_wf")
    workflow.connect([
        (inputnode, dual_echo_wf, [
            ("mask", "inputnode.mask"),
            ("echo_times", "inputnode.echo_times"),
            ("magnitude", "inputnode.magnitude"),
        ]),
        (calc_proposed_phase, dual_echo_wf, [("proposed_phase", "inputnode.phase")]),
    ])  # fmt:skip

    # Calculate a modified field map with 2pi added to the unwrapped difference image
    add_2pi = pe.Node(
        niu.Function(
            input_names=["phase", "unwrapped_diff", "echo_times"],
            output_names=["phase"],
            function=modify_unwrapped_diff,
        ),
        name="add_2pi",
    )
    workflow.connect([
        (inputnode, add_2pi, [
            ("phase", "phase"),
            ("echo_times", "echo_times"),
        ]),
        (unwrap_diffs, add_2pi, [("out_file", "unwrapped_diff")]),
    ])  # fmt:skip

    modified_dual_echo_wf = init_dual_echo_wf(name="modified_dual_echo_wf")
    workflow.connect([
        (inputnode, modified_dual_echo_wf, [
            ("mask", "inputnode.mask"),
            ("echo_times", "inputnode.echo_times"),
            ("magnitude", "inputnode.magnitude"),
        ]),
        (add_2pi, modified_dual_echo_wf, [("phase", "inputnode.phase")]),
    ])  # fmt:skip

    # Select the fieldmap
    select_unwrapped_diff = pe.Node(
        niu.Function(
            input_names=[
                "original_fieldmap",
                "original_unwrapped_phase",
                "original_offset",
                "modified_fieldmap",
                "modified_unwrapped_phase",
                "unwrapped_diff",
                "voxel_mask",
                "echo_times",
                "wrap_limit",
            ],
            output_names=["new_unwrapped_diff"],
            function=select_fieldmap,
        ),
        name="select_unwrapped_diff",
    )
    select_unwrapped_diff.inputs.wrap_limit = wrap_limit
    workflow.connect([
        (inputnode, select_unwrapped_diff, [("echo_times", "echo_times")]),
        (unwrap_diffs, select_unwrapped_diff, [("out_file", "unwrapped_diff")]),
        (create_mask, select_unwrapped_diff, [("mask", "voxel_mask")]),
        (calc_offset, select_unwrapped_diff, [("offset", "original_offset")]),
        (dual_echo_wf, select_unwrapped_diff, [
            ("outputnode.fieldmap", "original_fieldmap"),
            ("outputnode.unwrapped_phase", "original_unwrapped_phase"),
        ]),
        (modified_dual_echo_wf, select_unwrapped_diff, [
            ("outputnode.fieldmap", "modified_fieldmap"),
            ("outputnode.unwrapped_phase", "modified_unwrapped_phase"),
        ]),
        (select_unwrapped_diff, outputnode, [("new_unwrapped_diff", "unwrapped_diff")]),
    ])  # fmt:skip

    # Compute the updated phase offset
    calc_updated_offset = pe.Node(
        niu.Function(
            input_names=["phase", "unwrapped_diff", "echo_times"],
            output_names=["offset"],
            function=calculate_offset,
        ),
        name="calc_updated_offset",
    )
    workflow.connect([
        (inputnode, calc_updated_offset, [
            ("phase", "phase"),
            ("echo_times", "echo_times"),
        ]),
        (select_unwrapped_diff, calc_updated_offset, [("new_unwrapped_diff", "unwrapped_diff")]),
        (calc_updated_offset, outputnode, [("offset", "offset")]),
    ])  # fmt:skip

    return workflow


def init_dual_echo_wf(name="dual_echo_wf"):
    """Estimate a field map from the first two echoes of multi-echo data.

    Parameters
    ----------
    name : str
        The name of the workflow.

    Inputs
    ------
    magnitude : str
        The path to the magnitude image.
    phase : str
        The path to the phase image.
    echo_times : list of float
        The echo times of the multi-echo data.
    mask : str
        The path to the brain mask mask.

    Outputs
    -------
    fieldmap : str
        The path to the estimated fieldmap.
    unwrapped_phase : str
        The path to the unwrapped phase image.
    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "magnitude",
                "phase",
                "echo_times",
                "mask",
            ],
        ),
        name="inputnode",
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fieldmap", "unwrapped_phase"]),
        name="outputnode",
    )

    # Unwrap the phase with ROMEO
    unwrap_phase = pe.Node(
        ROMEO(
            weights="romeo",
            correct_global=True,
            max_seeds=1,
            merge_regions=False,
            correct_regions=False,
        ),
        name="unwrap_phase",
    )
    workflow.connect([
        (inputnode, unwrap_phase, [
            ("magnitude", "mag_file"),
            ("phase", "phase_file"),
            ("mask", "mask"),
            ("echo_times", "echo_times"),
        ]),
        (unwrap_phase, outputnode, [("out_file", "unwrapped_phase")]),
    ])  # fmt:skip

    # Calculate the fieldmap
    calc_fieldmap = pe.Node(
        niu.Function(
            input_names=["unwrapped_phase", "echo_times"],
            output_names=["fieldmap"],
            function=calculate_dual_echo_fieldmap,
        ),
        name="calc_fieldmap",
    )
    workflow.connect([
        (unwrap_phase, calc_fieldmap, [("out_file", "unwrapped_phase")]),
        (inputnode, calc_fieldmap, [("echo_times", "echo_times")]),
        (calc_fieldmap, outputnode, [("fieldmap", "fieldmap")]),
    ])  # fmt:skip

    return workflow
