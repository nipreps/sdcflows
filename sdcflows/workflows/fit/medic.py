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
"""MEDIC dynamic distortion correction (multi-echo phase + magnitude).

Backed by `warpkit <https://github.com/vanandrew/warpkit>`__, which is an
**optional** dependency carrying a Washington University non-commercial
license. Installing ``sdcflows[warpkit]`` opts the user into those terms;
the default ``sdcflows`` install never imports warpkit. Importing this
module does not require warpkit — the dependency is only resolved when the
:class:`~sdcflows.interfaces.warpkit.MEDIC` interface actually runs.
"""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

INPUT_FIELDS = ('phase', 'magnitude', 'metadata')

_MEDIC_DESC = """\
A dynamic *B<sub>0</sub>* nonuniformity map was estimated from multi-echo
magnitude and phase EPI series using MEDIC [Montez2024]_, as implemented in
``warpkit``.
"""


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
        Currently unused; reserved for future fast-path knobs.
    debug : :obj:`bool`
        Pass through to :class:`~sdcflows.interfaces.warpkit.MEDIC`.
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
        Static :math:`B_0` map in Hz (temporal mean of the dynamic series),
        for compatibility with the rest of sdcflows.
    fmap_dynamic : :obj:`str`
        4D :math:`B_0` map in Hz, one volume per timepoint (undistorted
        space). The actual MEDIC output; consumers wanting per-volume
        distortion correction should use this instead of ``fmap``.
    fmap_ref : :obj:`str`
        Brain-extracted magnitude reference (first echo, temporal mean).
    fmap_dynamic_ref : :obj:`str`
        4D first-echo magnitude (one volume per timepoint), for QC of
        per-volume correction. Untouched passthrough of the input.
    fmap_mask : :obj:`str`
        Static binary brain mask co-registered with ``fmap_ref``.
    fmap_dynamic_mask : :obj:`str`
        4D per-frame brain mask emitted by warpkit's phase-unwrapping
        stage; tracks subject motion frame-by-frame.
    fmap_coeff : :obj:`str` or :obj:`list` of :obj:`str`
        B-spline coefficients fit to the static ``fmap``. **Emitted as a
        compatibility shim**, not because it adds scientific value: the
        existing :func:`~sdcflows.workflows.apply.correction.init_unwarp_wf`
        consumes B-spline coefficients (via
        :class:`~sdcflows.interfaces.bspline.ApplyCoeffsField`), so MEDIC
        outputs need this representation to flow through the current apply
        pipeline. The spline fit is redundant for MEDIC — its fieldmap is
        already on the EPI grid (no resampling motivation) and warpkit's
        internal SVD filter already smooths each frame. A future
        MEDIC-aware apply workflow consuming ``fmap_dynamic`` directly
        should obsolete this output for MEDIC pipelines.
    method : :obj:`str`
        Short description string.

    """
    # Project-internal imports only — none of these load warpkit at module
    # import time. The warpkit dependency is resolved lazily inside
    # MEDIC._run_interface, so this workflow can be constructed (and the
    # module can be imported) without warpkit installed.
    from niworkflows.interfaces.images import IntraModalMerge

    from ...interfaces.bspline import DEFAULT_HF_ZOOMS_MM, BSplineApprox
    from ...interfaces.warpkit import ComputeFieldmap, UnwrapPhase
    from ..ancillary import init_brainextraction_wf

    workflow = Workflow(name=name)
    workflow.__desc__ = _MEDIC_DESC

    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                'fmap',
                'fmap_dynamic',
                'fmap_ref',
                'fmap_dynamic_ref',
                'fmap_mask',
                'fmap_dynamic_mask',
                'fmap_coeff',
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
    # both but hides the masks; we want them for ``fmap_dynamic_mask``.
    unwrap = pe.Node(UnwrapPhase(), name='unwrap', n_procs=omp_nthreads)
    unwrap.inputs.n_cpus = omp_nthreads
    unwrap.inputs.debug = debug

    compute_fmap = pe.Node(ComputeFieldmap(), name='compute_fmap', n_procs=omp_nthreads)
    compute_fmap.inputs.n_cpus = omp_nthreads

    # The 4D dynamic Hz fieldmap is the real MEDIC output — exposed below
    # as ``fmap_dynamic`` for consumers that want per-volume correction.
    # We also reduce it to 3D via temporal mean for the static ``fmap`` /
    # B-spline path, which the rest of sdcflows expects.
    fmap_mean = pe.Node(
        niu.Function(
            input_names=['in_file'],
            output_names=['out_file'],
            function=_temporal_mean,
        ),
        name='fmap_mean',
        run_without_submitting=True,
    )

    # First-echo magnitude — used (a) as the dynamic reference passthrough
    # and (b) temporally averaged for the static brain extraction.
    pick_mag1 = pe.Node(
        niu.Function(
            input_names=['in_list'], output_names=['out_file'], function=_first,
        ),
        name='pick_mag1',
        run_without_submitting=True,
    )
    magmrg = pe.Node(IntraModalMerge(hmc=False, to_ras=False), name='magmrg')
    brainextraction_wf = init_brainextraction_wf()

    # B-spline fit on the static fieldmap. Compatibility shim for the
    # existing init_unwarp_wf, which only consumes B-spline coefficients
    # (see ApplyCoeffsField). For MEDIC the spline adds no scientific
    # value — the fieldmap is already on the EPI grid and warpkit's SVD
    # filter has already smoothed it. See the ``fmap_coeff`` doc above.
    bs_filter = pe.Node(BSplineApprox(), name='bs_filter')
    bs_filter.interface._always_run = debug
    bs_filter.inputs.bs_spacing = [DEFAULT_HF_ZOOMS_MM]
    bs_filter.inputs.extrapolate = not debug
    if sloppy:
        bs_filter.inputs.zooms_min = 4.0

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
        (compute_fmap, fmap_mean, [('fieldmap', 'in_file')]),
        (compute_fmap, outputnode, [('fieldmap', 'fmap_dynamic')]),
        (unwrap, outputnode, [('masks', 'fmap_dynamic_mask')]),
        (inputnode, pick_mag1, [('magnitude', 'in_list')]),
        (pick_mag1, outputnode, [('out_file', 'fmap_dynamic_ref')]),
        (pick_mag1, magmrg, [('out_file', 'in_files')]),
        (magmrg, brainextraction_wf, [('out_avg', 'inputnode.in_file')]),
        (brainextraction_wf, outputnode, [
            ('outputnode.out_file', 'fmap_ref'),
            ('outputnode.out_mask', 'fmap_mask'),
        ]),
        (brainextraction_wf, bs_filter, [('outputnode.out_mask', 'in_mask')]),
        (fmap_mean, bs_filter, [('out_file', 'in_data')]),
        (bs_filter, outputnode, [
            ('out_extrapolated' if not debug else 'out_field', 'fmap'),
            ('out_coeff', 'fmap_coeff'),
        ]),
    ])
    # fmt: on

    return workflow


def _unpack_metadata(metadata):
    """Pull echo times (s→ms), TRT, and PE direction from BIDS sidecars."""
    if not metadata:
        raise ValueError('MEDIC requires per-echo metadata.')
    echo_times = [float(m['EchoTime']) * 1000.0 for m in metadata]
    total_readout_time = float(metadata[0]['TotalReadoutTime'])
    phase_encoding_direction = metadata[0]['PhaseEncodingDirection']
    peds = {m['PhaseEncodingDirection'] for m in metadata}
    if len(peds) > 1:
        raise ValueError(
            f'MEDIC echoes must share PhaseEncodingDirection; got {sorted(peds)}.'
        )
    return echo_times, total_readout_time, phase_encoding_direction


def _temporal_mean(in_file):
    """Average a 4D NIfTI across time, returning a 3D volume."""
    import os

    import nibabel as nb
    import numpy as np

    img = nb.load(in_file)
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data.mean(axis=3)
    out_file = os.path.abspath('fmap_mean.nii.gz')
    nb.Nifti1Image(data, img.affine, img.header).to_filename(out_file)
    return out_file


def _first(in_list):
    return in_list[0] if in_list else None
