1.4.0 (TBD)
===========
The *SDCFlows* 1.4.x series are release with a comprehensive overhaul of the software's API.
This overhaul has the vision of converting *SDCFlows* into some sort of subordinate pipeline
to other *d/fMRIPrep*, inline with *sMRIPrep*'s approach.
The idea is to consider fieldmaps a first-citizen input, for which derivatives are generated
at the output (on the same vein of, and effectively implementing `#26
<https://github.com/nipreps/sdcflows/issues/26>`__).
A bids's-eye view of this new release follows:

* Two new base objects (:py:class:`~sdcflows.fieldmaps.FieldmapFile` and
  :py:class:`~sdcflows.fieldmaps.FieldmapEstimation`) for the validation
  and representation of fieldmap estimation strategies.
  Validation of metadata and checking the sufficiency of imaging files
  and necessary parameters is now done with these two objects.
  :py:class:`~sdcflows.fieldmaps.FieldmapEstimation` also generates the
  appropriate estimation workflow for the input data.
* Moved estimation workflows under the :py:mod:`sdcflows.workflows.fit` module.
* New outputs submodule :py:mod:`sdcflows.workflows.outputs` that writes out reportlets and
  derivatives, following suit with higher-level *NiPreps* (*s/f/dMRIPrep*).
  The two workflows are exercised in the CircleCI tests, and the artifacts are generated
  this way.
  Derivatives are populated with relevant pieces of metadata (for instance, they forward
  the ``IntendedFor`` fields).
* A new :py:func:`~sdcflows.workflows.base.init_fmap_preproc_wf`, leveraging
  :py:class:`~sdcflows.fieldmaps.FieldmapEstimation` objects.
* Separated out a new utilities module :py:mod:`sdcflows.utils` for the manipulation of
  phase information and :abbr:`EPI (echo-planar imaging)` data.
* New :py:mod:`sdcflows.workflows.apply.registration` module, which aligns the reference map
  of the fieldmap of choice (e.g., a magnitude image) to the reference EPI
  (e.g., an SBRef, a *b=0* DWI, or a *fMRIPrep*'s *BOLDRef*) with ANTs.
  The workflow resamples the fieldmap reference into the reference EPI's space for
  reporting/visualization objectives.
* New :py:mod:`sdcflows.interfaces.bspline` set of utilities for the filtering and
  extrapolation of fieldmaps with B-Splines.
  Accordingly, all workflows have been updated to correctly handle (and better use) B-Spline
  coefficients.
* A new PEPOLAR implementation based on TOPUP (see
  :py:func:`sdcflows.workflows.fit.pepolar.init_topup_wf`).
* Pushed the code coverage with tests, along with a deep code cleanup.

Some of the most prominent pull-requests conducive to this release are:

* FIX: Convert SEI fieldmaps given in rad/s into Hz (#127)
* FIX: Limit ``3dQwarp`` to maximum 4 CPUs for stability reasons (#128)
* ENH: Finalizing the API overhaul (#132)
* ENH: Keep a registry of already-used identifiers (and auto-generate new) (#130)
* ENH: New objects for better representation of fieldmap estimation (#114)
* ENH: Show FieldmapReportlet oriented aligned with cardinal axes (#120)
* ENH: New estimation API (featuring a TOPUP implementation!) (#115)
* DOC: Enable NiPype's sphinx-extension to better render Interfaces (#131)
* MAINT: Minimal unit test and refactor of pepolar workflow node (#133)
* MAINT: Collect code coverage from tests on Circle (#122)
* MAINT: Test minimum dependencies with TravisCI (#96)
* MAINT: Add FLIRT config files to prepare for TOPUP integration (#116)

A complete list of issues addressed by the release is found `in the GitHub repo
<https://github.com/nipreps/sdcflows/milestone/2?closed=1>`__.

1.3.x series
============

1.3.3 (September 4, 2020)
-------------------------
Bug-fix release in 1.3.x series.

Allows niworkflows 1.2.x or 1.3.x, as no API-breaking changes in 1.3.0 affect SDCflows.

1.3.2 (August 14, 2020)
-----------------------
Bug-fix release in 1.3.x series.

* FIX: Replace NaNs in fieldmap atlas with zeros (#104)
* ENH: Return out_warp == "identity" if no SDC is applied (#108)

1.3.1 (May 22, 2020)
--------------------
Bug-fix release adapting to use newly refacored DerivativesDataSink

* ENH: Use new ``DerivativesDataSink`` from NiWorkflows 1.2.0 (#102)

1.3.0 (May 4, 2020)
-------------------
Minor release enforcing BIDS-Derivatives labels on ``dseg`` file.

* FIX: WM mask selection from dseg before generating report (#100)

Pre-1.3.x releases
==================

1.2.2 (April 16, 2020)
----------------------
Bug-fix release to fix phase-difference masking bug in the 1.2.x series.

* FIX: Do not reorient magnitude images (#98)

1.2.1 (April 01, 2020)
----------------------
A patch release to make *SDCFlows* more amicable to downstream software.

* MAINT: Migrate from versioneer to setuptools_scm (#97)
* MAINT: Flexibilize dependencies -- nipype, niworkflows, pybids (#95)

1.2.0 (February 15, 2020)
-------------------------
A minor version release that changes phasediff caclulations to improve robustness.
This release is preparation for *fMRIPrep* 20.0.0.

* FIX: Scale all phase maps to ``[0, 2pi]`` range (#88)
* MNT: Fix package tests (#90)
* MNT: Fix circle deployment (#91)

1.1.0 (February 3, 2020)
------------------------
This is a nominal release that enables downstream tools to depend on both
SDCFlows and niworkflows 1.1.x.

Bug fixes needed for the 1.5.x series of fMRIPrep will be accepted into the
1.0.x series of SDCFlows.

1.0.6 (April 15, 2020)
----------------------
Bug-fix release.

* FIX: Do not reorient magnitude images (#98)

1.0.5 (February 14, 2020)
-------------------------
Bug-fix release.

* FIX: Center phase maps around central mode, avoiding FoV-related outliers (#89)

1.0.4 (January 27, 2020)
------------------------
Bug-fix release.

* FIX: Connect SyN outputs whenever SyN is run (#82)
* MNT: Skim Docker image, optimize CircleCI workflow, and reuse cached results (#80)

1.0.3 (December 18, 2019)
-------------------------
A hotfix release preventing downstream dependency collisions on fMRIPrep.

* PIN: niworkflows-1.0.3 `449c2c2
  <https://github.com/nipreps/sdcflows/commit/449c2c2b0ab383544f5024de82ca8a80ee70894d>`__

1.0.2 (December 18, 2019)
-------------------------
A hotfix release.

* FIX: NiWorkflows' ``IntraModalMerge`` choked with images of shape (x, y, z, 1) (#79, `2e6faa0
  <https://github.com/nipreps/sdcflows/commit/2e6faa05ed0f0ec0b4616f33db778a61a1df89d0>`__,
  `717a69e
  <https://github.com/nipreps/sdcflows/commit/717a69ef680d556e4d5cde6876d0e60b023924e0>`__,
  and `361cd67
  <https://github.com/nipreps/sdcflows/commit/361cd678215fca9434baa713fa43f77a2231e632>`__)

1.0.1 (December 04, 2019)
-------------------------
A bugfix release.

* FIX: Flexibly and cheaply select initial PEPOLAR volumes (#75)
* ENH: Phase1/2 - subtract phases before unwrapping (#70)

1.0.0 (November 25, 2019)
-------------------------
A first stable release after detaching these workflows off from *fMRIPrep*.

With thanks to Matthew Cieslak and Azeez Adebimpe.

* FIX: Hard-wire ``MNI152NLin2009cAsym`` as standard space for SDC-SyN (#63)
* ENH: Base implementation for phase1/2 fieldmaps (#60)
* ENH: Update ``spatialimage.get_data()`` -> ``spatialimage.get_fdata()`` (#58)
* ENH: Refactor fieldmap-unwarping flows, more homogeneous interface (#56)
* ENH: Transparency on fieldmap plots! (#57)
* ENH: Stop using siemens2rads from old nipype workflows (#50)
* ENH: Large refactor of the orchestration workflow (#55)
* ENH: Refactor the distortion estimation workflow (#53)
* ENH: Deduplicating magnitude handling and fieldmap postprocessing workflows (#52)
* ENH: Do not use legacy demean function from old nipype workflows (#51)
* ENH: Revise and add tests for the PEPOLAR correction (#29)
* ENH: Improved fieldmap reportlets (#28)
* ENH: Set-up testing framework (#27)
* DOC: Update documentation (#61)
* DOC: Fix typo and link to BIDS Specification (#49)
* DOC: Build API documentation (#43)
* CI: Add check to avoid deployment of documentation from forks (#48)
* CI: Fix CircleCI builds by adding a [refresh workdir] commit message tag (#47)
* CI: Optimize CircleCI using a local docker registry instead docker save/load (#45)
* MAINT: Housekeeping - flake8 errors, settings, etc. (#44)
* MAINT: Rename boldrefs to distortedrefs (#41)
* MAINT: Use niflow-nipype1-workflows for old nipype.workflows imports (#39)

0.1.4 (November 22, 2019)
-------------------------
A maintenance release to pin niworkflows to version 1.0.0rc1.

0.1.3 (October 15, 2019)
------------------------
Adapts *SDCflows* to the separation of workflows from Nipype 1.

* MAINT: pin `niflow-nipype1-workflows`, `nipype` and update corresponding imports.

0.1.2 (October 10, 2019)
------------------------
BAD RELEASE -- DO NOT USE

0.1.1 (July 23, 2019)
---------------------
Minor fixup of the deploy infrastructure from CircleCI

* MAINT: Add manifest including versioneer (#25) @effigies

0.1.0 (July 22, 2019)
---------------------
First version working with *fMRIPrep* v1.4.1.
