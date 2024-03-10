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

INPUT_FIELDS = ("magnitude", "phase", "metadata")


def init_medic_wf(
    automask,
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
            ]
        ),
        name="outputnode",
    )
    outputnode.inputs.method = "MEDIC"

    # Convert phase to radians (-pi to pi)
    phase2rad = pe.MapNode(
        PhaseMap2rads(),
        iterfield=["in_file"],
        name="phase2rad",
    )
    workflow.connect([(inputnode, phase2rad, [("phase", "in_file")])])

    # Split phase data into single-frame, multi-echo 4D files
    split_phase = pe.MapNode(
        FSLSplit(dimension="t"),
        iterfield=["in_file"],
        name="split_phase",
    )
    workflow.connect([(phase2rad, split_phase, [("out_file", "in_file")])])

    concat_across_echoes = pe.Node(
        niu.Merge(2),
        name="concat_across_echoes",
    )
    workflow.connect([(split_phase, concat_across_echoes, [("out_files", "in1")])])

    if automask:
        # the theory goes like this, the magnitude/otsu base mask can be too aggressive
        # occasionally and the voxel quality mask can get extra voxels that are not brain,
        # but is noisy so we combine the two masks to get a better mask

        # Use ROMEO's voxel-quality command

        # Then use skimage's otsu thresholding to get a mask
        # and do a bunch of other stuff
        ...

    # Do MCPC-3D-S algo to compute phase offset

    # Unwrap phase data with ROMEO
    unwrap_phase = pe.MapNode(
        ROMEO(),
        name="unwrap_phase",
    )
    workflow.connect([(concat_across_echoes, unwrap_phase, [("out", "in_file")])])

    # Re-split the unwrapped phase data

    # Re-combine into echo-wise time series

    # Check temporal consistency of phase unwrapping

    # Compute field maps

    # Apply SVD filter to field maps

    return workflow
