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
r"""
Processing phase-difference and *directly measured* :math:`B_0` maps.

Theory
~~~~~~
The displacement suffered by every voxel along the phase-encoding (PE) direction
can be derived from Eq. (2) of [Hutton2002]_:

.. math::

    \Delta_\text{PE} (i, j, k) = \gamma \cdot \Delta B_0 (i, j, k) \cdot T_\text{ro},
    \label{eq:fieldmap-1}\tag{1}

where
:math:`\Delta_\text{PE} (i, j, k)` is the *voxel-shift map* (VSM) along the *PE* direction,
:math:`\gamma` is the gyromagnetic ratio of the H proton in Hz/T
(:math:`\gamma = 42.576 \cdot 10^6 \, \text{Hz} \cdot \text{T}^\text{-1}`),
:math:`\Delta B_0 (i, j, k)` is the *fieldmap variation* in T (Tesla), and
:math:`T_\text{ro}` is the readout time of one slice of the EPI dataset
we want to correct for distortions.

Let :math:`V` represent the «*fieldmap in Hz*» (or equivalently,
«*voxel-shift-velocity map*» as Hz are equivalent to voxels/s), with
:math:`V(i,j,k) = \gamma \cdot \Delta B_0 (i, j, k)`, then, introducing
the voxel zoom along the phase-encoding direction, :math:`s_\text{PE}`,
we obtain the nonzero component of the associated displacements field
:math:`\Delta D_\text{PE} (i, j, k)` that unwarps the target EPI dataset:

.. math::

    \Delta D_\text{PE} (i, j, k) = V(i, j, k) \cdot T_\text{ro} \cdot s_\text{PE}.
    \label{eq:fieldmap-2}\tag{2}


.. _sdc_direct_b0 :

Direct B0 mapping sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Some MR schemes such as :abbr:`SEI (spiral-echo imaging)` can directly
reconstruct an estimate of *the fieldmap in Hz*, :math:`V(i,j,k)`.
These *fieldmaps* are described with more detail `here
<https://cni.stanford.edu/wiki/GE_Processing#Fieldmaps>`__.

This corresponds to `this section of the BIDS specification
<https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#case-3-direct-field-mapping>`__.

.. _sdc_phasediff :

Phase-difference B0 estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The fieldmap variation in T, :math:`\Delta B_0 (i, j, k)`, that is necessary to obtain
:math:`\Delta_\text{PE} (i, j, k)` in Eq. :math:`\eqref{eq:fieldmap-1}` can be
calculated from two subsequient :abbr:`GRE (Gradient-Recalled Echo)` echoes,
via eq. (1) of [Hutton2002]_:

.. math::

    \Delta B_0 (i, j, k) = \frac{\Delta \Theta (i, j, k)}{2\pi \cdot \gamma \, \Delta\text{TE}},
    \label{eq:fieldmap-3}\tag{3}

where
:math:`\Delta \Theta (i, j, k)` is the phase-difference map in radians,
and :math:`\Delta\text{TE}` is the elapsed time between the two GRE echoes.

For simplicity, the «*voxel-shift-velocity map*» :math:`V(i,j,k)`, which we
can introduce in Eq. :math:`\eqref{eq:fieldmap-2}` to directly obtain
the displacements field, can be obtained as:

.. math::

    V(i, j, k) = \frac{\Delta \Theta (i, j, k)}{2\pi \cdot \Delta\text{TE}}.
    \label{eq:fieldmap-4}\tag{4}

This calculation is further complicated by the fact that :math:`\Theta_i`
(and therefore, :math:`\Delta \Theta`) are clipped (or *wrapped*) within
the range :math:`[0 \dotsb 2\pi )`.
It is necessary to find the integer number of offsets that make a region
continuously smooth with its neighbors (*phase-unwrapping*, [Jenkinson2003]_).

This corresponds to `this section of the BIDS specification
<https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#two-phase-images-and-two-magnitude-images>`__.
Some scanners produce one ``phasediff`` map, where the drift between the two echos has
already been calculated (see `the corresponding section of BIDS
<https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#case-1-phase-difference-map-and-at-least-one-magnitude-image>`__).

References
----------
.. [Hutton2002] Hutton et al., Image Distortion Correction in fMRI: A Quantitative
    Evaluation, NeuroImage 16(1):217-240, 2002. doi:`10.1006/nimg.2001.1054
    <https://doi.org/10.1006/nimg.2001.1054>`__.
.. [Jenkinson2003] Jenkinson, M. (2003) Fast, automated, N-dimensional phase-unwrapping
    algorithm. MRM 49(1):193-197. doi:`10.1002/mrm.10354
    <https://doi.org/10.1002/mrm.10354>`__.

"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

INPUT_FIELDS = ("magnitude", "fieldmap")


def init_fmap_wf(omp_nthreads=1, sloppy=False, debug=False, mode="phasediff", name="fmap_wf"):
    """
    Estimate the fieldmap based on a field-mapping MRI acquisition.

    Estimates the fieldmap using either one phase-difference map or
    image and one or more
    magnitude images corresponding to two or more :abbr:`GRE (Gradient Echo sequence)`
    acquisitions.

    When we have a sequence that directly measures the fieldmap,
    we just need to mask it (using the corresponding magnitude image)
    to remove the noise in the surrounding air region, and ensure that
    units are Hz.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.fieldmap import init_fmap_wf
            wf = init_fmap_wf(omp_nthreads=6)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use.
    sloppy : :obj:`bool`
        Whether a fast but less accurate correction should be applied.
    debug : :obj:`bool`
        Run in debug mode
    name : :obj:`str`
        Unique name of this workflow.

    Inputs
    ------
    magnitude : :obj:`list` of :obj:`str`
        Path to the corresponding magnitude image for anatomical reference.
    fieldmap : :obj:`list` of :obj:`tuple`(:obj:`str`, :obj:`dict`)
        Path to the fieldmap acquisition (``*_fieldmap.nii[.gz]`` of BIDS).

    Outputs
    -------
    fmap : :obj:`str`
        Path to the estimated fieldmap.
    fmap_ref : :obj:`str`
        Path to a preprocessed magnitude image reference.
    fmap_coeff : :obj:`str` or :obj:`list` of :obj:`str`
        The path(s) of the B-Spline coefficients supporting the fieldmap.
    fmap_mask : :obj:`str`
        Path to a binary brain mask corresponding to the ``fmap`` and ``fmap_ref``
        pair.
    method: :obj:`str`
        Short description of the estimation method that was run.

    """
    from ...interfaces.bspline import (
        BSplineApprox,
        DEFAULT_LF_ZOOMS_MM,
        DEFAULT_HF_ZOOMS_MM,
        DEFAULT_ZOOMS_MM,
    )

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=INPUT_FIELDS), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fmap", "fmap_ref", "fmap_mask", "fmap_coeff", "method"]),
        name="outputnode",
    )

    magnitude_wf = init_magnitude_wf(omp_nthreads=omp_nthreads)
    bs_filter = pe.Node(BSplineApprox(), n_procs=omp_nthreads, name="bs_filter")
    bs_filter.interface._always_run = debug
    bs_filter.inputs.bs_spacing = (
        [DEFAULT_LF_ZOOMS_MM, DEFAULT_HF_ZOOMS_MM] if not sloppy else [DEFAULT_ZOOMS_MM]
    )
    bs_filter.inputs.extrapolate = not debug

    # fmt: off
    workflow.connect([
        (inputnode, magnitude_wf, [("magnitude", "inputnode.magnitude")]),
        (magnitude_wf, bs_filter, [("outputnode.fmap_mask", "in_mask")]),
        (magnitude_wf, outputnode, [
            ("outputnode.fmap_mask", "fmap_mask"),
            ("outputnode.fmap_ref", "fmap_ref"),
        ]),
        (bs_filter, outputnode, [
            ("out_extrapolated" if not debug else "out_field", "fmap"),
            ("out_coeff", "fmap_coeff")]),
    ])
    # fmt: on

    if mode == "phasediff":
        workflow.__postdesc__ = """\
A *B<sub>0</sub>* nonuniformity map (or *fieldmap*) was estimated from the
phase-drift map(s) measure with two consecutive GRE (gradient-recalled echo)
acquisitions.
"""
        phdiff_wf = init_phdiff_wf(omp_nthreads, debug=debug)
        outputnode.inputs.method = "FMB (fieldmap-based) - phase-difference map"

        # fmt: off
        workflow.connect([
            (inputnode, phdiff_wf, [("fieldmap", "inputnode.phase")]),
            (magnitude_wf, phdiff_wf, [
                ("outputnode.fmap_ref", "inputnode.magnitude"),
                ("outputnode.fmap_mask", "inputnode.mask"),
            ]),
            (phdiff_wf, bs_filter, [
                ("outputnode.fieldmap", "in_data"),
            ]),
        ])
        # fmt: on
    else:
        from niworkflows.interfaces.images import IntraModalMerge
        from ...interfaces.fmap import CheckB0Units

        workflow.__postdesc__ = """\
A *B<sub>0</sub>* nonuniformity map (or *fieldmap*) was directly measured with
an MRI scheme designed with that purpose such as SEI (Spiral-Echo Imaging).
"""
        outputnode.inputs.method = "FMB (fieldmap-based) - directly measured B0 map"
        # Merge input fieldmap images (assumes all are given in the same units!)
        fmapmrg = pe.Node(
            IntraModalMerge(zero_based_avg=False, hmc=False, to_ras=False),
            name="fmapmrg",
        )
        units = pe.Node(CheckB0Units(), name="units", run_without_submitting=True)

        # fmt: off
        workflow.connect([
            (inputnode, units, [(("fieldmap", _get_units), "units")]),
            (inputnode, fmapmrg, [(("fieldmap", _get_file), "in_files")]),
            (fmapmrg, units, [("out_avg", "in_file")]),
            (units, bs_filter, [("out_file", "in_data")]),
        ])
        # fmt: on

    return workflow


def init_magnitude_wf(omp_nthreads, name="magnitude_wf"):
    """
    Prepare the magnitude part of :abbr:`GRE (gradient-recalled echo)` fieldmaps.

    Average (if not done already) the magnitude part of the
    :abbr:`GRE (gradient recalled echo)` images, run N4 to
    correct for B1 field nonuniformity, and skull-strip the
    preprocessed magnitude.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.fieldmap import init_magnitude_wf
            wf = init_magnitude_wf(omp_nthreads=6)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``magnitude_wf``)

    Inputs
    ------
    magnitude : :obj:`os.PathLike`
        Path to the corresponding magnitude path(s).

    Outputs
    -------
    fmap_ref : :obj:`os.PathLike`
        Path to the fieldmap reference calculated in this workflow.
    fmap_mask : :obj:`os.PathLike`
        Path to a binary brain mask corresponding to the reference above.

    """
    from niworkflows.interfaces.images import IntraModalMerge
    from ..ancillary import init_brainextraction_wf

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["magnitude"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fmap_ref", "fmap_mask", "mask_report"]),
        name="outputnode",
    )

    # Merge input magnitude images
    # Do not reorient to RAS to preserve the validity of PhaseEncodingDirection
    magmrg = pe.Node(IntraModalMerge(hmc=False, to_ras=False), name="magmrg")
    brainextraction_wf = init_brainextraction_wf()

    # fmt: off
    workflow.connect([
        (inputnode, magmrg, [("magnitude", "in_files")]),
        (magmrg, brainextraction_wf, [("out_avg", "inputnode.in_file")]),
        (brainextraction_wf, outputnode, [
            ("outputnode.out_file", "fmap_ref"),
            ("outputnode.out_mask", "fmap_mask"),
            ("outputnode.out_probseg", "fmap_probseg"),
        ]),
    ])
    # fmt: on
    return workflow


def init_phdiff_wf(omp_nthreads, debug=False, name="phdiff_wf"):
    r"""
    Generate a :math:`B_0` field from consecutive-phases and phase-difference maps.

    This workflow preprocess phase-difference maps (or generates the phase-difference
    map should two ``phase1``/``phase2`` be provided at the input), and generates
    an image equivalent to BIDS's ``fieldmap`` that can be processed with the
    general fieldmap workflow.

    Besides phase2 - phase1 subtraction, the core of this particular workflow relies
    in the phase-unwrapping with FSL PRELUDE [Jenkinson2003]_.
    FSL PRELUDE takes wrapped maps in the range 0 to 6.28, `as per the user guide
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#Step_2_-_Getting_.28wrapped.29_phase_in_radians>`__.

    For the phase-difference maps, recentering back to :math:`[-\pi \dotsb \pi )`
    is necessary.
    After some massaging and with the scaling of the echo separation factor
    :math:`\Delta \text{TE}`, the phase-difference maps are converted into
    an actual :math:`B_0` map in Hz units.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.fieldmap import init_phdiff_wf
            wf = init_phdiff_wf(omp_nthreads=1)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    debug : :obj:`bool`
        Run in debug mode
    name : :obj:`str`
        Name of workflow (default: ``phdiff_wf``)

    Inputs
    ------
    magnitude : :obj:`os.PathLike`
        A reference magnitude image preprocessed elsewhere.
    phase : :obj:`list` of :obj:`tuple` of (:obj:`os.PathLike`, :obj:`dict`)
        List containing one GRE phase-difference map with its corresponding metadata
        (requires ``EchoTime1`` and ``EchoTime2``), or the phase maps for the two
        subsequent echoes, with their metadata (requires ``EchoTime``).
    mask : :obj:`os.PathLike`
        A brain mask calculated from the magnitude image.

    Outputs
    -------
    fieldmap : :obj:`os.PathLike`
        The estimated fieldmap in Hz.

    """
    from nipype.interfaces.fsl import PRELUDE
    from ...interfaces.fmap import Phasediff2Fieldmap, PhaseMap2rads, SubtractPhases

    workflow = Workflow(name=name)
    workflow.__postdesc__ = f"""\
The corresponding phase-map(s) were phase-unwrapped with `prelude` (FSL {PRELUDE.version}).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["magnitude", "phase", "mask"]), name="inputnode"
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fieldmap"]),
        name="outputnode",
    )

    def _split(phase):
        return phase

    split = pe.MapNode(  # We cannot use an inline connection function with MapNode
        niu.Function(function=_split, output_names=["map_file", "meta"]),
        iterfield=["phase"],
        run_without_submitting=True,
        name="split",
    )

    # phase diff -> radians
    phmap2rads = pe.MapNode(
        PhaseMap2rads(),
        name="phmap2rads",
        iterfield=["in_file"],
        run_without_submitting=True,
    )
    # FSL PRELUDE will perform phase-unwrapping
    prelude = pe.Node(PRELUDE(), name="prelude")

    calc_phdiff = pe.Node(
        SubtractPhases(), name="calc_phdiff", run_without_submitting=True
    )
    calc_phdiff.interface._always_run = debug
    compfmap = pe.Node(Phasediff2Fieldmap(), name="compfmap")

    # fmt: off
    workflow.connect([
        (inputnode, split, [("phase", "phase")]),
        (inputnode, prelude, [("magnitude", "magnitude_file"),
                              ("mask", "mask_file")]),
        (split, phmap2rads, [("map_file", "in_file")]),
        (phmap2rads, calc_phdiff, [("out_file", "in_phases")]),
        (split, calc_phdiff, [("meta", "in_meta")]),
        (calc_phdiff, prelude, [("phase_diff", "phase_file")]),
        (prelude, compfmap, [("unwrapped_phase_file", "in_file")]),
        (calc_phdiff, compfmap, [("metadata", "metadata")]),
        (compfmap, outputnode, [("out_file", "fieldmap")]),
    ])
    # fmt: on
    return workflow


def _get_file(intuple):
    """
    Extract the filename from the inputnode.

    >>> _get_file([("fmap.nii.gz", {"Units": "rad/s"})])
    'fmap.nii.gz'

    >>> _get_file(("fmap.nii.gz", {"Units": "rad/s"}))
    'fmap.nii.gz'

    """
    if isinstance(intuple, list):
        intuple = intuple[0]
    return intuple[0]


def _get_units(intuple):
    """
    Extract Units from metadata.

    >>> _get_units([("fmap.nii.gz", {"Units": "rad/s"})])
    'rad/s'

    >>> _get_units(("fmap.nii.gz", {"Units": "rad/s"}))
    'rad/s'

    """
    if isinstance(intuple, list):
        intuple = intuple[0]
    return intuple[1]["Units"]
