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
"""
Align the fieldmap reference map to the target EPI.

The fieldmap reference map may be a magnitude image (or an EPI dataset,
in the case of PEPOLAR estimation).

The target EPI is the distorted dataset (or a reference thereof).

"""
from warnings import warn
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

from ... import data


def init_coeff2epi_wf(
    omp_nthreads,
    sloppy=False,
    debug=False,
    write_coeff=False,
    name="coeff2epi_wf",
):
    """
    Move the field coefficients on to the target (distorted) EPI space.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.apply.registration import init_coeff2epi_wf
            wf = init_coeff2epi_wf(omp_nthreads=2)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use.
    sloppy : :obj:`bool`
        Run fast configurations of registrations.
    debug : :obj:`bool`
        Run in debug mode, that is, generating additional traces of performance.
    name : :obj:`str`
        Unique name of this workflow.
    write_coeff : :obj:`bool`
        Map coefficients file

    Inputs
    ------
    target_ref
        the target EPI reference image
    target_mask
        the reference image (skull-stripped)
    fmap_ref
        the reference (anatomical) image corresponding to ``fmap``
    fmap_mask
        a brain mask corresponding to ``fmap``
    fmap_coeff
        fieldmap coefficients

    Outputs
    -------
    fmap_coeff
        fieldmap coefficients in the space of the target reference EPI
    target_ref
        the target reference EPI resampled into the fieldmap reference for
        quality control purposes.
    target2fmap_xfm
        An ITK-style transform produced with ANTs that can be used as transform in
        ``antsApplyTransforms`` with the fieldmap as reference and the target EPI
        as moving, resampling the latter into the fieldmap space.

    """
    from packaging.version import parse as parseversion, Version
    from niworkflows.interfaces.fixes import FixHeaderRegistration as Registration

    from sdcflows.interfaces import brainmask

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
The estimated *fieldmap* was then aligned with rigid-registration to the target
EPI (echo-planar imaging) reference run.
The field coefficients were mapped on to the reference EPI using the transform.
"""
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=["target_ref", "target_mask", "fmap_ref", "fmap_mask", "fmap_coeff"]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["target_ref", "fmap_coeff", "target2fmap_xfm"]),
        name="outputnode",
    )

    # Dilate only for coregistration purposes
    # https://github.com/nipreps/sdcflows/issues/461
    dilate_target_mask = pe.Node(brainmask.BinaryDilation(radius=5), name="dilate_target_mask")
    dilate_fmap_mask = pe.Node(brainmask.BinaryDilation(radius=5), name="dilate_fmap_mask")

    # Register the reference of the fieldmap to the reference
    # of the target image (the one that shall be corrected)
    coregister = pe.Node(
        Registration(
            from_file=data.load(f"fmap-any_registration{'_testing' * sloppy}.json"),
            output_warped_image=debug,
            output_inverse_warped_image=debug,
        ),
        name="coregister",
        n_procs=omp_nthreads,
    )

    ver = coregister.interface.version or "2.2.0"
    mask_trait_s = "s" if parseversion(ver) >= Version("2.2.0") else ""

    # fmt: off
    workflow.connect([
        (inputnode, outputnode, [("fmap_coeff", "fmap_coeff")]),
        (inputnode, dilate_target_mask, [("target_mask", "in_file")]),
        (inputnode, dilate_fmap_mask, [("fmap_mask", "in_file")]),
        (inputnode, coregister, [
            ("target_ref", "moving_image"),
            ("fmap_ref", "fixed_image"),
        ]),
        (dilate_target_mask, coregister, [
            ("out_file", f"moving_image_mask{mask_trait_s}")
        ]),
        (dilate_fmap_mask, coregister, [
            ("out_file", f"fixed_image_mask{mask_trait_s}")
        ]),
        (coregister, outputnode, [
            ("warped_image", "target_ref"),
            ("forward_transforms", "target2fmap_xfm"),
        ]),
    ])
    # fmt: on

    if write_coeff:
        warn(
            "SDCFlows does not tinker with the coefficients file anymore, "
            "the `write_coeff` parameter will be removed in a future release."
        )

    return workflow
