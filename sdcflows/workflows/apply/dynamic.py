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
    jacobian=True,
    omp_nthreads=1,
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
    jacobian : :obj:`bool`
        If :obj:`True`, apply Jacobian determinant correction after
        resampling, preserving total signal through compression/expansion
        regions of the EPI distortion. Mirrors the
        :func:`~sdcflows.workflows.apply.correction.init_unwarp_wf` default.
    omp_nthreads : :obj:`int`
        Per-node ``n_procs`` hint for the Nipype scheduler. Note that
        warpkit's ``apply_warp`` / ``convert_fieldmap`` C++ paths don't
        accept a thread count, so this only affects scheduling, not
        warpkit's internal parallelism.
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

    # No coregistration step: warpkit emits ``fmap_dynamic`` on the EPI grid
    # by construction (the fieldmap is computed from the same multi-echo
    # acquisition being corrected here), so the static path's
    # ``fmap2data_xfm`` plumbing has no analog.

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

    # Optional Jacobian-determinant intensity correction. Mirrors what
    # ``init_unwarp_wf`` does via ``ApplyCoeffsField(jacobian=True)`` —
    # we just call the same numpy formula (``transform.fieldmap_jacobian``)
    # against the dynamic 4D fieldmap, post-resampling.
    if jacobian:
        jac_correct = pe.Node(
            niu.Function(
                input_names=['in_file', 'fmap_dynamic', 'pe_direction', 'readout_time'],
                output_names=['out_file'],
                function=_apply_jacobian,
            ),
            name='jac_correct',
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
        (average, brainextraction_wf, [('out_file', 'inputnode.in_file')]),
        (convert_fmap, outputnode, [('out_file', 'fieldwarp')]),
        (brainextraction_wf, outputnode, [
            ('outputnode.out_file', 'corrected_ref'),
            ('outputnode.out_mask', 'corrected_mask'),
        ]),
    ])
    # fmt: on

    if jacobian:
        # fmt: off
        workflow.connect([
            (apply_warp, jac_correct, [('out_file', 'in_file')]),
            (inputnode, jac_correct, [('fmap_dynamic', 'fmap_dynamic')]),
            (rotime, jac_correct, [('pe_direction', 'pe_direction'),
                                   ('readout_time', 'readout_time')]),
            (jac_correct, average, [('out_file', 'in_file')]),
            (jac_correct, outputnode, [('out_file', 'corrected')]),
        ])
        # fmt: on
    else:
        # fmt: off
        workflow.connect([
            (apply_warp, average, [('out_file', 'in_file')]),
            (apply_warp, outputnode, [('out_file', 'corrected')]),
        ])
        # fmt: on

    return workflow


def _apply_jacobian(in_file, fmap_dynamic, pe_direction, readout_time):
    """Multiply ``in_file`` by the per-frame Jacobian determinant of ``fmap_dynamic``.

    ``in_file`` and ``fmap_dynamic`` must share the spatial grid (the
    dynamic apply path guarantees this — ``fmap_dynamic`` is on the EPI
    grid by construction). The PE-axis sign convention matches warpkit's
    ``ConvertFieldmap``: ``pe_direction`` ending in ``-`` flips the
    readout time, the result is fed to
    :func:`~sdcflows.transform.fieldmap_jacobian`.
    """
    import os

    import nibabel as nb
    import numpy as np

    from sdcflows.transform import fieldmap_jacobian

    pe_axis = 'ijk'.index(pe_direction[0])
    ro_signed = -float(readout_time) if pe_direction.endswith('-') else float(readout_time)

    img = nb.load(in_file)
    fmap_img = nb.load(fmap_dynamic)
    data = np.asanyarray(img.dataobj, dtype='float32')
    fmap_hz = np.asanyarray(fmap_img.dataobj, dtype='float32')

    jac = fieldmap_jacobian(fmap_hz, ro_signed, pe_axis)
    corrected = data * jac

    out_file = os.path.abspath('corrected_jac.nii.gz')
    nb.Nifti1Image(corrected.astype('float32'), img.affine, img.header).to_filename(out_file)
    return out_file
