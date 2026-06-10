# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""MEDIC dynamic distortion correction (multi-echo phase + magnitude).

Backed by `warpkit <https://github.com/vanandrew/warpkit>`__, which is an
**optional** dependency carrying a Washington University non-commercial
license. Installing ``sdcflows[warpkit]`` opts the user into those terms;
the default ``sdcflows`` install never imports warpkit. Importing this
module does not require warpkit — the dependency is only resolved when the
:class:`~sdcflows.interfaces.warpkit.UnwrapPhase` and
:class:`~sdcflows.interfaces.warpkit.ComputeFieldmap` interfaces actually run.
"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

INPUT_FIELDS = ('phase', 'magnitude', 'metadata')


def init_medic_wf(
    omp_nthreads=1,
    sloppy=False,
    debug=False,
    name='medic_wf',
    **kwargs,
):
    """
    Estimate a fieldmap via MEDIC from multi-echo magnitude + phase EPI.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.medic import init_medic_wf
            wf = init_medic_wf()

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads warpkit may use.
    sloppy : :obj:`bool`
        Accepted for parity with other ``init_*_wf`` constructors; currently
        unused for MEDIC.
    debug : :obj:`bool`
        Pass through to :class:`~sdcflows.interfaces.warpkit.UnwrapPhase`.
    name : :obj:`str`
        Workflow name.

    Inputs
    ------
    phase : :obj:`list` of :obj:`str`
        Phase NIfTI per echo.
    magnitude : :obj:`list` of :obj:`str`
        Magnitude NIfTI per echo.
    metadata : :obj:`list` of :obj:`dict`
        BIDS sidecar dicts, one per echo. Must contain ``EchoTime``,
        ``TotalReadoutTime``, and ``PhaseEncodingDirection``.

    Outputs
    -------
    fmap : :obj:`str`
        4D :math:`B_0` map in Hz, one volume per timepoint, already on the
        EPI grid. Consumers must dispatch on dimensionality (3D for static
        estimators, 4D for MEDIC) when applying.
    fmap_ref : :obj:`str`
        First-echo magnitude series, unprocessed: one volume per timepoint
        matching ``fmap``.
    fmap_mask : :obj:`str`
        4D binary brain mask (one mask per timepoint) as produced by MEDIC
        (``warpkit``), aligned with ``fmap``.
    method : :obj:`str`
        Short description string.

    """
    # Project-internal imports only — none of these load warpkit at module
    # import time. The warpkit dependency is resolved lazily inside the
    # MEDIC interfaces at run time.
    from ...interfaces.warpkit import ComputeFieldmap, UnwrapPhase

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
A dynamic *B<sub>0</sub>* nonuniformity map was estimated from multi-echo
magnitude and phase EPI series using MEDIC [@van2026medic], as implemented in
``warpkit``.
"""

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'fmap',
                'fmap_ref',
                'fmap_mask',
                'method',
            ],
        ),
        name='outputnode',
    )
    outputnode.inputs.method = 'MEDIC (multi-echo dynamic distortion correction)'

    # Pull echo_times / TRT / PED from sidecar dicts so warpkit gets them as
    # direct args. (The interfaces also accept JSON sidecar paths, but the
    # upstream sdcflows layer passes dicts.)
    extract_meta = pe.Node(
        niu.Function(
            input_names=['metadata'],
            output_names=['echo_times', 'total_readout_time', 'phase_encoding_direction'],
            function=_unpack_metadata,
        ),
        name='extract_meta',
        run_without_submitting=True,
    )

    # Two-stage warpkit path: UnwrapPhase exposes per-frame masks, which
    # ComputeFieldmap then consumes. The one-shot MEDIC interface bundles
    # both but doesn't materially differ for the fieldmap outputs we need.
    unwrap = pe.Node(
        UnwrapPhase(n_cpus=omp_nthreads, debug=debug),
        name='unwrap',
        n_procs=omp_nthreads,
    )

    # ComputeFieldmap doesn't expose a ``debug`` input — only UnwrapPhase
    # does, so the asymmetry is intentional.
    compute_fmap = pe.Node(
        ComputeFieldmap(n_cpus=omp_nthreads),
        name='compute_fmap',
        n_procs=omp_nthreads,
    )

    # ``fmap_ref`` is just the first-echo magnitude series, passed through
    # untouched. ``fmap_mask`` reuses the per-frame masks MEDIC already
    # computes during phase unwrapping, so both track the per-volume fieldmap
    # without any extra N4/skull-strip work.
    pick_mag1 = pe.Node(
        niu.Function(
            input_names=['in_list'],
            output_names=['out_file'],
            function=_first,
        ),
        name='pick_mag1',
        run_without_submitting=True,
    )

    # fmt: off
    workflow.connect([
        (inputnode, extract_meta, [('metadata', 'metadata')]),
        (inputnode, unwrap, [('phase', 'phase'),
                             ('magnitude', 'magnitude')]),
        (extract_meta, unwrap, [('echo_times', 'echo_times')]),
        (inputnode, compute_fmap, [('magnitude', 'magnitude')]),
        (unwrap, compute_fmap, [('unwrapped', 'unwrapped'),
                                ('masks', 'masks')]),
        (extract_meta, compute_fmap, [
            ('echo_times', 'echo_times'),
            ('total_readout_time', 'total_readout_time'),
            ('phase_encoding_direction', 'phase_encoding_direction'),
        ]),
        (compute_fmap, outputnode, [('fieldmap', 'fmap')]),
        (inputnode, pick_mag1, [('magnitude', 'in_list')]),
        (pick_mag1, outputnode, [('out_file', 'fmap_ref')]),
        (unwrap, outputnode, [('masks', 'fmap_mask')]),
    ])
    # fmt: on

    return workflow


def _unpack_metadata(metadata):
    """Pull echo times (s→ms), TRT, and PE direction from BIDS sidecars."""
    if not metadata:
        raise ValueError('MEDIC requires per-echo metadata.')
    if len(metadata) < 2:
        raise ValueError(
            f'MEDIC requires at least two echoes; got {len(metadata)}. '
            '(FieldmapEstimation enforces this for wrangler-built workflows; '
            'this guard catches direct callers that bypass it.)'
        )
    echo_times = [float(m['EchoTime']) * 1000.0 for m in metadata]
    total_readout_time = float(metadata[0]['TotalReadoutTime'])
    phase_encoding_direction = metadata[0]['PhaseEncodingDirection']
    peds = {m['PhaseEncodingDirection'] for m in metadata}
    if len(peds) > 1:
        raise ValueError(f'MEDIC echoes must share PhaseEncodingDirection; got {sorted(peds)}.')
    return echo_times, total_readout_time, phase_encoding_direction


def _first(in_list):
    return in_list[0] if in_list else None
