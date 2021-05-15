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
"""Estimate fieldmaps for :abbr:`SDC (susceptibility distortion correction)`."""
from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

LOGGER = logging.getLogger("nipype.workflow")


def init_brainextraction_wf(name="brainextraction_wf"):
    """
    Remove nonbrain tissue from images.

    Parameters
    ----------
    name : :obj:`str`, optional
        Workflow name (default: ``"brainextraction_wf"``)

    Inputs
    ------
    in_file : :obj:`str`
        the GRE magnitude or EPI reference to be brain-extracted
    bspline_dist : :obj:`int`, optional
        Integer to replace default distance of b-spline separation for N4

    Outputs
    -------
    out_file : :obj:`str`
        the input file after N4 and smart clipping
    out_brain : :obj:`str`
        the output file, just the brain extracted
    out_mask : :obj:`str`
        the calculated mask
    out_probseg : :obj:`str`
        a probability map that the random walker reached
        a given voxel (some sort of "soft" brainmask)

    """
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from niworkflows.interfaces.nibabel import IntensityClip
    from ..interfaces.brainmask import BrainExtraction

    wf = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=("in_file", "bspline_dist")), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=(
                "out_file",
                "out_brain",
                "out_mask",
                "out_probseg",
            )
        ),
        name="outputnode",
    )
    clipper_pre = pe.Node(IntensityClip(), name="clipper_pre")

    # de-gradient the fields ("bias/illumination artifact")
    n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            shrink_factor=4,
        ),
        n_procs=8,
        name="n4",
    )
    clipper_post = pe.Node(IntensityClip(p_min=0.01, p_max=99.9), name="clipper_post")
    masker = pe.Node(BrainExtraction(), name="masker")

    # fmt:off
    wf.connect([
        (inputnode, clipper_pre, [("in_file", "in_file")]),
        (inputnode, n4, [("bspline_dist", "bspline_fitting_distance")]),
        (clipper_pre, n4, [("out_file", "input_image")]),
        (n4, clipper_post, [("output_image", "in_file")]),
        (clipper_post, masker, [("out_file", "in_file")]),
        (clipper_post, outputnode, [("out_file", "out_file")]),
        (masker, outputnode, [("out_file", "out_brain"),
                              ("out_mask", "out_mask"),
                              ("out_probseg", "out_probseg")]),
    ])
    # fmt:on

    return wf
