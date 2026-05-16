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
"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

INPUT_FIELDS = ('distorted', 'metadata', 'fmap')


def init_dynamic_unwarp_wf(
    *,
    jacobian=True,
    omp_nthreads=1,
    name='dynamic_unwarp_wf',
):
    r"""
    Apply a per-volume 4D fieldmap to unwarp a 4D EPI series.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.apply.dynamic import init_dynamic_unwarp_wf
            wf = init_dynamic_unwarp_wf()

    Parameters
    ----------
    jacobian : :obj:`bool`
        If :obj:`True`, apply Jacobian determinant correction after
        resampling, preserving total signal through compression/expansion
        regions of the EPI distortion. Mirrors the
        :func:`~sdcflows.workflows.apply.correction.init_unwarp_wf` default.
    omp_nthreads : :obj:`int`
        Maximum number of parallel volume resamplings.
    name : :obj:`str`
        Workflow name.

    Inputs
    ------
    distorted : :obj:`str`
        4D EPI series to unwarp. Must share frame count with ``fmap``.
    metadata : :obj:`dict`
        BIDS sidecar metadata. ``TotalReadoutTime`` and
        ``PhaseEncodingDirection`` are required.
    fmap : :obj:`str`
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
    """
    from niworkflows.interfaces.images import RobustAverage

    from ...interfaces.epi import GetReadoutTime
    from ..ancillary import init_brainextraction_wf

    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['corrected', 'corrected_ref', 'corrected_mask'],
        ),
        name='outputnode',
    )

    rotime = pe.Node(GetReadoutTime(), name='rotime', run_without_submitting=True)

    # No coregistration step: the dynamic fieldmap is on the EPI grid by
    # construction (e.g., warpkit's MEDIC output is computed from the same
    # multi-echo acquisition being corrected here), so the static path's
    # ``fmap2data_xfm`` plumbing has no analog.
    unwarp = pe.Node(
        niu.Function(
            input_names=[
                'distorted',
                'fmap',
                'pe_direction',
                'readout_time',
                'jacobian',
                'num_threads',
            ],
            output_names=['out_file'],
            function=_dynamic_unwarp,
        ),
        name='unwarp',
        n_procs=omp_nthreads,
    )
    unwarp.inputs.jacobian = jacobian
    unwarp.inputs.num_threads = omp_nthreads

    average = pe.Node(RobustAverage(mc_method=None), name='average')
    brainextraction_wf = init_brainextraction_wf()

    # fmt: off
    workflow.connect([
        (inputnode, rotime, [('distorted', 'in_file'),
                             ('metadata', 'metadata')]),
        (inputnode, unwarp, [('distorted', 'distorted'),
                             ('fmap', 'fmap')]),
        (rotime, unwarp, [('pe_direction', 'pe_direction'),
                          ('readout_time', 'readout_time')]),
        (unwarp, average, [('out_file', 'in_file')]),
        (unwarp, outputnode, [('out_file', 'corrected')]),
        (average, brainextraction_wf, [('out_file', 'inputnode.in_file')]),
        (brainextraction_wf, outputnode, [
            ('outputnode.out_file', 'corrected_ref'),
            ('outputnode.out_mask', 'corrected_mask'),
        ]),
    ])
    # fmt: on

    return workflow


def _dynamic_unwarp(distorted, fmap, pe_direction, readout_time, jacobian, num_threads):
    """Resample a 4D EPI through a per-frame 4D Hz fieldmap on the same grid.

    Uses :func:`~sdcflows.transform.apply_dynamic_unwarp`, which delegates
    per-volume resampling to the same scipy-backed primitives that
    :class:`~sdcflows.transform.B0FieldTransform` uses for the static path.
    """
    import os

    from sdcflows.transform import apply_dynamic_unwarp

    resampled = apply_dynamic_unwarp(
        distorted,
        fmap,
        pe_dir=pe_direction,
        ro_time=readout_time,
        jacobian=jacobian,
        num_threads=num_threads,
    )
    out_file = os.path.abspath('corrected.nii.gz')
    resampled.to_filename(out_file)
    return out_file
