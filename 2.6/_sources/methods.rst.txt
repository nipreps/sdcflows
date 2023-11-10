Methods and implementation
==========================
*SDCFlows* defines a clear :abbr:`API (application programming interface)` that divides
the problem of susceptibility distortions (SD) into two stages:

#. **Estimation**:
   the MRI acquisitions in the protocol for :abbr:`SD (susceptibility distortions)` are
   discovered and preprocessed to estimate a map of B\ :sub:`0` non-uniformity in Hz (:math:`\Delta B_0`).
   The theory behind these distortions is well described in the literature ([Jezzard1995]_, [Hutton2002]_),
   and further discussed below (see a summary in :numref:`fig-1`).
   *SDCFlows* builds on freely-available software to implement three major strategies for estimating
   :math:`\Delta B_0` (Eq. :math:`\eqref{eq:fieldmap-1}`).
   These strategies are described below, and implemented within :py:mod:`sdcflows.workflows.fit`\ .

#. **Application**:
   the B-Spline basis coefficients used to represent the estimated :math:`\Delta B_0` map mapped into the
   target :abbr:`EPI (echo-planar imaging)` scan to be corrected, and a displacement field in NIfTI
   format that is compatible with ANTs is interpolated from the B-Spline basis.
   The voxel location error along the :abbr:`PE (phase-encoding)` will be proportional to :math:`\Delta B_0 \cdot T_\text{ro}`,
   where :math:`T_\text{ro}` is the *total readout time* of the target :abbr:`EPI (echo-planar imaging)` (:numref:`fig-1`).
   The implementation of these workflows is found in the submodule :py:mod:`sdcflows.workflows.apply`\ .

.. _fig-1:

.. figure:: _static/sdcflows-OHBM21-fig1.png
   :width: 100%
   :align: center

   Susceptibility distortions in a nutshell

.. admonition:: BIDS Specification

    See the section `Echo-planar imaging and *B0* mapping
    <https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#echo-planar-imaging-and-b0-mapping>`__.

Fieldmap estimation: theory and methods
---------------------------------------
The displacement suffered by every voxel along the :abbr:`PE (phase-encoding)` direction
can be derived from Eq. (2) of [Hutton2002]_:

.. math::

    \Delta_\text{PE} (i, j, k) = \gamma \cdot \Delta B_0 (i, j, k) \cdot T_\text{ro},
    \label{eq:fieldmap-1}\tag{1}

where
:math:`\Delta_\text{PE} (i, j, k)` is the *voxel-shift map* (VSM) along the :abbr:`PE (phase-encoding)` direction,
:math:`\gamma` is the gyromagnetic ratio of the H proton in Hz/T
(:math:`\gamma = 42.576 \cdot 10^6 \, \text{Hz} \cdot \text{T}^\text{-1}`),
:math:`\Delta B_0 (i, j, k)` is the *fieldmap variation* in T (Tesla), and
:math:`T_\text{ro}` is the readout time of one slice of the :abbr:`EPI (echo-planar imaging)` dataset
we want to correct for distortions.

Let :math:`V` represent the *fieldmap* in Hz (or equivalently,
*voxel-shift-velocity map*, as Hz are equivalent to voxels/s), with
:math:`V(i,j,k) = \gamma \cdot \Delta B_0 (i, j, k)`, then, introducing
the voxel zoom along the phase-encoding direction, :math:`s_\text{PE}`,
we obtain the nonzero component of the associated displacements field
:math:`\Delta D_\text{PE} (i, j, k)` that unwarps the target :abbr:`EPI (echo-planar imaging)` dataset:

.. math::

    \Delta D_\text{PE} (i, j, k) = V(i, j, k) \cdot T_\text{ro} \cdot s_\text{PE}.
    \label{eq:fieldmap-2}\tag{2}

.. _sdc_direct_b0 :

Direct B0 mapping sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. admonition:: BIDS Specification

    See the section `Types of fieldmaps - Case 3: Direct field mapping
    <https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#case-3-direct-field-mapping>`__
    in the BIDS specification.

Some MR schemes such as :abbr:`SEI (spiral-echo imaging)` can directly
reconstruct an estimate of *the fieldmap in Hz*, :math:`V(i,j,k)`.
These *fieldmaps* are described with more detail `here
<https://cni.stanford.edu/wiki/GE_Processing#Fieldmaps>`__.

.. _sdc_phasediff :

Phase-difference B0 estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. admonition:: BIDS Specification

    See the section `Types of fieldmaps - Case 2: Two phase maps and two magnitude images
    <https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#case-2-two-phase-maps-and-two-magnitude-images>`__
    in the BIDS specification.

    Some scanners produce one ``phasediff`` map, where the drift between the two echos has
    already been calculated, see the section
    `Types of fieldmaps - Case 1: Phase-difference map and at least one magnitude image
    <https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#case-1-phase-difference-map-and-at-least-one-magnitude-image>`__.

The fieldmap variation in T, :math:`\Delta B_0 (i, j, k)`, that is necessary to obtain
:math:`\Delta_\text{PE} (i, j, k)` in Eq. :math:`\eqref{eq:fieldmap-1}` can be
calculated from two subsequent :abbr:`GRE (Gradient-Recalled Echo)` echoes,
via Eq. (1) of [Hutton2002]_:

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

.. _sdc_pepolar :

:abbr:`PEPOLAR (Phase Encoding POLARity)` techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. admonition:: BIDS Specification

    See the section `Types of fieldmaps - Case 4: Multiple phase encoded directions ("pepolar")
    <https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#case-4-multiple-phase-encoded-directions-pepolar>`__.

Alternatively, it is possible to estimate the field by exploiting the symmetry of the
distortion when the :abbr:`PE (phase-encoding)` polarity is reversed.
*SDCFlows* integrates two implementations based on FSL ``topup`` [Andersson2003]_,
and AFNI ``3dQwarp`` [Cox1997]_.
By default, FSL ``topup`` will be used.

.. _sdc_fieldmapless :

Fieldmap-less approaches
~~~~~~~~~~~~~~~~~~~~~~~~
Many studies acquired (especially with legacy MRI protocols) do not have any
information to estimate susceptibility-derived distortions.
In the absence of data with the specific purpose of estimating the :math:`B_0`
inhomogeneity map, researchers resort to nonlinear registration to an
«*anatomically correct*» map of the same individual (normally acquired with
:abbr:`T1w (T1-weighted)`, or :abbr:`T2w (T2-weighted)` sequences).
One of the most prominent proposals of this approach is found in [Studholme2000]_.

*SDCFlows* includes an (experimental) procedure, based on nonlinear image registration
with ANTs' symmetric normalization (SyN) technique.
This workflow takes a skull-stripped :abbr:`T1w (T1-weighted)` image and
a reference :abbr:`EPI (Echo-Planar Imaging)` image, and estimates a field of nonlinear
displacements that accounts for susceptibility-derived distortions.
To more accurately estimate the warping on typically distorted regions, this
implementation uses an average :math:`B_0` mapping described in [Treiber2016]_.
The implementation is a variation on those developed in [Huntenburg2014]_ and
[Wang2017]_.

The process is divided in two steps.
First, the two images to be aligned (anatomical and one or more :abbr:`EPI (echo-planar imaging)` sources) are prepared for
registration, including a linear pre-alignment of both, calculation of a 3D :abbr:`EPI (echo-planar imaging)` *reference* map,
intensity/histogram enhancement, and *deobliquing* (meaning, for images where the physical
coordinates axes and the data array axes are not aligned, the physical coordinates are
rotated to align with the data array).
Such a preprocessing is implemented in
:py:func:`~sdcflows.workflows.fit.syn.init_syn_preprocessing_wf`.
Second, the outputs of the preprocessing workflow are fed into
:py:func:`~sdcflows.workflows.fit.syn.init_syn_sdc_wf`,
which executes the nonlinear, SyN registration.
To aid the *Mattes* mutual information cost function, the registration scheme is set up
in *multi-channel* mode, and laplacian-filtered derivatives of both anatomical and :abbr:`EPI (echo-planar imaging)`
reference are introduced as a second registration channel.
The optimization gradients of the registration process are weighted, so that deformations
effectively possible only along the :abbr:`PE (phase-encoding)` axis.
Given that ANTs' registration framework performs on physical coordinates, it is necessary
that input images are not *oblique*.
The anatomical image is used as *fixed image*, and therefore, the registration process
estimates the transformation function from *unwarped* (anatomically *correct*) coordinates
to *distorted* coordinates.
If fed into ``antsApplyTransforms``, the resulting transform will effectively *unwarp* a distorted
:abbr:`EPI (echo-planar imaging)` given as input into its *unwarped* mapping.
The estimated transform is then converted into a :math:`B_0` fieldmap in Hz, which can be
stored within the derivatives folder.

.. danger ::

    This procedure is experimental, and the outcomes should be scrutinized one-by-one
    and used with caution.
    Feedback will be enthusiastically received.

Other (unsupported) approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There exist some alternative options to estimate the fieldmap, such as mapping the
point-spread-function [Zaitsev2004]_, or by means of nonlinear registration of brain
surfaces onto the distorted :abbr:`EPI (echo-planar imaging)` data [Esteban2016]_.

Estimation tooling
~~~~~~~~~~~~~~~~~~
The workflows provided by :py:mod:`sdcflows.workflows.fit` make use of several utilities.
The cornerstone of these tools is the fieldmap representation with B-Splines
(:py:mod:`sdcflows.interfaces.bspline`).
B-Splines are well-suited to plausibly smooth the fieldmap and provide a compact
representation of the field with fewer parameters.
This representation is also more accurate in the case the images that were used for estimation
are not aligned with the target images to be corrected because the fieldmap is not directly
interpolated in the projection, but rather, the position of the coefficients in space is
updated and then the fieldmap reconstructed.

Unwarping the distorted data
----------------------------
:py:mod:`sdcflows.workflows.apply` contains workflows to project fieldmaps represented by B-Spline
basis into the space of the target :abbr:`EPI (echo-planar imaging)` data.

Discovering fieldmaps in a BIDS dataset
---------------------------------------
To ease the implementation of higher-level pipelines integrating :abbr:`SDC (susceptibility distortion correction)`,
*SDCFlows* provides :py:func:`sdcflows.utils.wrangler.find_estimators`.

Explicit specification with ``B0FieldIdentifier``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. admonition:: BIDS Specification

    See the section `Expressing the MR protocol intent for fieldmaps - Using *B0FieldIdentifier* metadata
    <https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#using-b0fieldidentifier-metadata>`__.

If one or more ``B0FieldIdentifier``\ (s) are set within the input metadata (following BIDS' specifications),
then corresponding estimators will be built from the available input data.

Implicit specification with ``IntendedFor``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. admonition:: BIDS Specification

    See the section `Expressing the MR protocol intent for fieldmaps - Using *IntendedFor* metadata
    <https://bids-specification.readthedocs.io/en/latest/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#using-intendedfor-metadata>`__.

In the case no ``B0FieldIdentifier``\ (s) are defined, then *SDCFlows* will try to identify as many fieldmap
estimators as possible within the dataset following a set of heuristics.
Then, those estimators may be linked to target :abbr:`EPI (echo-planar imaging)` data by interpreting the
``IntendedFor`` field if available.

Fieldmap-less by registration to a T1-weighted image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If none of the two previous options yielded any workable estimation strategy, and provided that
the argument ``fmapless`` is set to ``True``, then :py:func:`~sdcflows.utils.wrangler.find_estimators`
will attempt to find :abbr:`BOLD (blood-oxygen level-dependent)` or :abbr:`DWI (diffusion-weighted imaging)`
instances within single sessions that are consistent in :abbr:`PE (phase-encoding)` direction and
*total readout time*, assuming they have been acquired with the same shimming settings.

If one or more anatomical images are found, and if the search for candidate
:abbr:`BOLD (blood-oxygen level-dependent)` or :abbr:`DWI (diffusion-weighted imaging)` data
yields results, then corresponding fieldmap-less estimators are set up.

It is possible to force the fieldmap-less estimation by passing ``force_fmapless=True`` to the
:py:func:`~sdcflows.utils.wrangler.find_estimators` utility.

References
----------
.. [Jezzard1995] Jezzard, P. & Balaban, R. S. (1995) Correction for geometric distortion in
    echo planar images from B0 field variations. Magn. Reson. Med. 34:65–73.
    doi:`10.1002/mrm.1910340111 <https://doi.org/10.1002/mrm.1910340111>`__.
.. [Hutton2002] Hutton et al., (2002) Image Distortion Correction in fMRI: A Quantitative
    Evaluation, NeuroImage 16(1):217-240. doi:`10.1006/nimg.2001.1054
    <https://doi.org/10.1006/nimg.2001.1054>`__.
.. [Jenkinson2003] Jenkinson, M. (2003) Fast, automated, N-dimensional phase-unwrapping
    algorithm. MRM 49(1):193-197. doi:`10.1002/mrm.10354
    <https://doi.org/10.1002/mrm.10354>`__.
.. [Andersson2003] Andersson, J. (2003) How to correct susceptibility distortions in spin-echo
    echo-planar images: application to diffusion tensor imaging. NeuroImage 20:870–888.
    doi:`10.1016/s1053-8119(03)00336-7 <https://doi.org/10.1016/s1053-8119(03)00336-7>`__.
.. [Cox1997] Cox, R. (1997) Software tools for analysis and visualization of fMRI data. NMR Biomed.
    10:171–178, doi:`10.1002/(sici)1099-1492(199706/08)10:4/5%3C171::aid-nbm453%3E3.0.co;2-l
    <https://doi.org/10.1002/(sici)1099-1492(199706/08)10:4/5%3C171::aid-nbm453%3E3.0.co;2-l>`__.
.. [Studholme2000] Studholme et al. (2000) Accurate alignment of functional EPI data to
    anatomical MRI using a physics-based distortion model,
    IEEE Trans Med Imag 19(11):1115-1127, 2000, doi: `10.1109/42.896788
    <https://doi.org/10.1109/42.896788>`__.
.. [Treiber2016] Treiber, J. M. et al. (2016) Characterization and Correction
    of Geometric Distortions in 814 Diffusion Weighted Images,
    PLoS ONE 11(3): e0152472. doi:`10.1371/journal.pone.0152472
    <https://doi.org/10.1371/journal.pone.0152472>`_.
.. [Wang2017] Wang S, et al. (2017) Evaluation of Field Map and Nonlinear
    Registration Methods for Correction of Susceptibility Artifacts
    in Diffusion MRI. Front. Neuroinform. 11:17.
    doi:`10.3389/fninf.2017.00017
    <https://doi.org/10.3389/fninf.2017.00017>`_.
.. [Huntenburg2014] Huntenburg, J. M. (2014) `Evaluating Nonlinear
    Coregistration of BOLD EPI and T1w Images
    <http://pubman.mpdl.mpg.de/pubman/item/escidoc:2327525:5/component/escidoc:2327523/master_thesis_huntenburg_4686947.pdf>`__,
    Berlin: Master Thesis, Freie Universität.
.. [Zaitsev2004] Zaitsev, M. (2004) Point spread function mapping with parallel imaging techniques and
    high acceleration factors: Fast, robust, and flexible method for echo-planar imaging distortion correction,
    MRM 52(5):1156-1166. doi:`10.1002/mrm.20261 <https://doi.org/10.1002/mrm.20261>`__.
.. [Esteban2016] Esteban, O. (2016) Surface-driven registration method for the structure-informed segmentation
    of diffusion MR images. NeuroImage 139:450-461.
    doi:`10.1016/j.neuroimage.2016.05.011 <https://doi.org/10.1016/j.neuroimage.2016.05.011>`__.
