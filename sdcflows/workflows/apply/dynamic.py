# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2025 The NiPreps Developers <nipreps@gmail.com>
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
"""Per-volume distortion correction using a dynamic (4D) fieldmap.

Counterpart to :func:`~sdcflows.workflows.apply.correction.init_unwarp_wf`,
specialized for fieldmaps that vary across time (typically MEDIC). Where
the static apply path interpolates a single B-spline-encoded field onto
the EPI grid and applies the same warp to every volume, this workflow
takes a 4D Hz fieldmap *already on the EPI grid* and applies a different
warp to each timepoint.

Backed by ``warpkit``, an optional dependency. The workflow is
module-load pure: warpkit is only resolved when the underlying interfaces
actually run.
"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

INPUT_FIELDS = ('distorted', 'metadata', 'fmap_dynamic')


def init_dynamic_unwarp_wf(
    *,
    omp_nthreads=1,
    debug=False,
    name='dynamic_unwarp_wf',
):
    r"""
    Apply a per-volume MEDIC fieldmap to unwarp a 4D EPI series.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.apply.dynamic import init_dynamic_unwarp_wf
            wf = init_dynamic_unwarp_wf()

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads warpkit may use.
    debug : :obj:`bool`
        Reserved.
    name : :obj:`str`
        Workflow name.

    Inputs
    ------
    distorted : :obj:`str`
        4D EPI series to unwarp. Must share frame count with ``fmap_dynamic``.
    metadata : :obj:`dict`
        BIDS sidecar metadata. ``TotalReadoutTime`` and
        ``PhaseEncodingDirection`` are required.
    fmap_dynamic : :obj:`str`
        4D B\ :sub:`0` field map in Hz, already on the EPI grid (typically
        from :func:`~sdcflows.workflows.fit.medic.init_medic_wf`).

    Outputs
    -------
    corrected : :obj:`str`
        4D unwarped EPI.
    corrected_ref : :obj:`str`
        3D temporal-mean reference of the corrected series, brain-extracted.
    corrected_mask : :obj:`str`
        Binary brain mask co-registered with ``corrected_ref``.
    fieldwarp : :obj:`str`
        4D displacement map (mm along PE axis) used for the resampling.
    """
    # Project-internal imports only; warpkit stays unloaded until interfaces run.
    from niworkflows.interfaces.images import RobustAverage

    from ...interfaces.epi import GetReadoutTime
    from ...interfaces.warpkit import ApplyWarp, ConvertFieldmap
    from ..ancillary import init_brainextraction_wf

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['corrected', 'corrected_ref', 'corrected_mask', 'fieldwarp'],
        ),
        name='outputnode',
    )

    rotime = pe.Node(GetReadoutTime(), name='rotime', run_without_submitting=True)

    # Hz fieldmap → 1-channel mm displacement map along the PE axis.
    convert_fmap = pe.Node(
        ConvertFieldmap(from_type='fieldmap', to_type='map'),
        name='convert_fmap',
        n_procs=omp_nthreads,
    )

    # Per-frame resampling. Frame count of `transform` must match `distorted`.
    apply_warp = pe.Node(
        ApplyWarp(transform_type='map'),
        name='apply_warp',
        n_procs=omp_nthreads,
    )

    average = pe.Node(RobustAverage(mc_method=None), name='average')
    brainextraction_wf = init_brainextraction_wf()

    # fmt: off
    workflow.connect([
        (inputnode, rotime, [('distorted', 'in_file'),
                             ('metadata', 'metadata')]),
        (rotime, convert_fmap, [('pe_direction', 'phase_encoding_direction'),
                                ('readout_time', 'total_readout_time')]),
        (inputnode, convert_fmap, [('fmap_dynamic', 'in_file')]),
        (rotime, apply_warp, [('pe_direction', 'phase_encoding_axis')]),
        (convert_fmap, apply_warp, [('out_file', 'transform')]),
        (inputnode, apply_warp, [('distorted', 'in_file')]),
        (apply_warp, average, [('out_file', 'in_file')]),
        (average, brainextraction_wf, [('out_file', 'inputnode.in_file')]),
        (apply_warp, outputnode, [('out_file', 'corrected')]),
        (convert_fmap, outputnode, [('out_file', 'fieldwarp')]),
        (brainextraction_wf, outputnode, [
            ('outputnode.out_file', 'corrected_ref'),
            ('outputnode.out_mask', 'corrected_mask'),
        ]),
    ])
    # fmt: on

    return workflow
