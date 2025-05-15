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
"""Applying a fieldmap given its B-Spline coefficients in Hz."""
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def init_unwarp_wf(
    *,
    jacobian=True,
    use_metadata_estimates=False,
    fallback_total_readout_time=None,
    free_mem=None,
    omp_nthreads=1,
    debug=False,
    name="unwarp_wf",
):
    r"""
    Set up a workflow that unwarps the input :abbr:`EPI (echo-planar imaging)` dataset.

    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.apply.correction import init_unwarp_wf
            wf = init_unwarp_wf(omp_nthreads=2)

    Parameters
    ----------
    jacobian : :class:`bool`
        If :obj:`True`, apply Jacobian determinant correction after unwarping.
    omp_nthreads : :class:`int`
        Maximum number of threads an individual process may use.
    name : :class:`str`
        Unique name of this workflow.
    debug : :class:`bool`
        Whether to run in *sloppy* mode.

    Inputs
    ------
    distorted
        the target EPI image.
    metadata
        dictionary of metadata corresponding to the target EPI image
    fmap_coeff
        fieldmap coefficients in distorted EPI space.
    hmc_xforms
        list of head-motion correction matrices (in ITK format)

    Outputs
    -------
    fieldmap
        the actual B\ :sub:`0` inhomogeneity map (also called *fieldmap*)
        interpolated from the B-Spline coefficients into the target EPI's
        grid, given in Hz units.
    fieldwarp
        the displacements field interpolated from the B-Spline coefficients
        and scaled by the appropriate parameters (readout time of the EPI
        target and voxel size along PE).
    corrected
        the target EPI reference image, after applying unwarping.
    corrected_mask
        a fast mask calculated from the corrected EPI reference.

    """
    from niworkflows.interfaces.images import RobustAverage
    from niworkflows.interfaces.nibabel import MergeSeries
    from sdcflows.interfaces.epi import GetReadoutTime
    from sdcflows.interfaces.bspline import ApplyCoeffsField
    from sdcflows.workflows.ancillary import init_brainextraction_wf
    from sdcflows.utils.misc import front as _pop

    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "distorted",
                "metadata",
                "fmap_coeff",
                "fmap2data_xfm",
                "data2fmap_xfm",
                "hmc_xforms",
            ]
        ),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "fieldmap",
                "fieldwarp",
                "corrected",
                "corrected_ref",
                "corrected_mask",
            ]
        ),
        name="outputnode",
    )

    rotime = pe.Node(
        GetReadoutTime(
            use_estimate=use_metadata_estimates,
        ),
        name="rotime",
        run_without_submitting=True,
    )
    rotime.interface._always_run = debug
    if fallback_total_readout_time is not None:
        rotime.inputs.fallback = fallback_total_readout_time

    # resample is memory-hungry; choose a smaller number of threads
    # if we know how much memory we have to work with
    mem_per_thread = 5  # True for a 128x128x84 image; should generalize
    if debug:
        num_threads = 1
    elif free_mem is not None:
        mem_gb = min(0.9 * free_mem, mem_per_thread * omp_nthreads)
        num_threads = max(int(mem_gb // mem_per_thread), 1)
    else:
        num_threads = omp_nthreads

    resample = pe.Node(
        ApplyCoeffsField(jacobian=jacobian, num_threads=num_threads),
        mem_gb=mem_per_thread * num_threads,
        name="resample",
    )

    merge = pe.Node(MergeSeries(), name="merge")
    average = pe.Node(RobustAverage(mc_method=None), name="average")

    brainextraction_wf = init_brainextraction_wf()

    # fmt:off
    workflow.connect([
        (inputnode, rotime, [(("distorted", _pop), "in_file"),
                             ("metadata", "metadata")]),
        (inputnode, resample, [("distorted", "in_data"),
                               ("fmap_coeff", "in_coeff"),
                               ("fmap2data_xfm", "fmap2data_xfm"),
                               ("data2fmap_xfm", "data2fmap_xfm"),
                               ("hmc_xforms", "in_xfms")]),
        (rotime, resample, [("readout_time", "ro_time"),
                            ("pe_direction", "pe_dir")]),
        (resample, merge, [("out_corrected", "in_files")]),
        (merge, average, [("out_file", "in_file")]),
        (average, brainextraction_wf, [("out_file", "inputnode.in_file")]),
        (merge, outputnode, [("out_file", "corrected")]),
        (resample, outputnode, [("out_field", "fieldmap")]),
        (brainextraction_wf, outputnode, [
            ("outputnode.out_file", "corrected_ref"),
            ("outputnode.out_mask", "corrected_mask"),
        ]),
    ])
    # fmt:on
    return workflow
