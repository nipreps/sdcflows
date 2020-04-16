1.2.2 (April 16, 2020)
======================
Bug-fix release to fix phase-difference masking bug in the 1.2.x series.

* FIX: Do not reorient magnitude images (#98)

1.0.6 (April 15, 2020)
======================
Bug-fix release.

* FIX: Do not reorient magnitude images (#98)

1.2.1 (April 01, 2020)
======================
A patch release to make *SDCFlows* more amicable to downstream software.

* MAINT: Migrate from versioneer to setuptools_scm (#97)
* MAINT: Flexibilize dependencies -- nipype, niworkflows, pybids (#95)

1.2.0 (February 15, 2020)
=========================
A minor version release that changes phasediff caclulations to improve robustness.
This release is preparation for *fMRIPrep* 20.0.0.

* FIX: Scale all phase maps to ``[0, 2pi]`` range (#88)
* MNT: Fix package tests (#90)
* MNT: Fix circle deployment (#91)

1.0.5 (February 14, 2020)
=========================
Bug-fix release.

* FIX: Center phase maps around central mode, avoiding FoV-related outliers (#89)

1.1.0 (February 3, 2020)
========================
This is a nominal release that enables downstream tools to depend on both
SDCFlows and niworkflows 1.1.x.

Bug fixes needed for the 1.5.x series of fMRIPrep will be accepted into the
1.0.x series of SDCFlows.

1.0.4 (January 27, 2020)
=========================
Bug-fix release.

* FIX: Connect SyN outputs whenever SyN is run (#82)
* MNT: Skim Docker image, optimize CircleCI workflow, and reuse cached results (#80)

1.0.3 (December 18, 2019)
=========================
A hotfix release preventing downstream dependency collisions on fMRIPrep.

* PIN: niworkflows-1.0.3 `449c2c2
  <https://github.com/nipreps/sdcflows/commit/449c2c2b0ab383544f5024de82ca8a80ee70894d>`__

1.0.2 (December 18, 2019)
=========================
A hotfix release.

* FIX: NiWorkflows' ``IntraModalMerge`` choked with images of shape (x, y, z, 1) (#79, `2e6faa0
  <https://github.com/nipreps/sdcflows/commit/2e6faa05ed0f0ec0b4616f33db778a61a1df89d0>`__,
  `717a69e
  <https://github.com/nipreps/sdcflows/commit/717a69ef680d556e4d5cde6876d0e60b023924e0>`__,
  and `361cd67
  <https://github.com/nipreps/sdcflows/commit/361cd678215fca9434baa713fa43f77a2231e632>`__)

1.0.1 (December 04, 2019)
=========================
A bugfix release.

* FIX: Flexibly and cheaply select initial PEPOLAR volumes (#75)
* ENH: Phase1/2 - subtract phases before unwrapping (#70)

1.0.0 (November 25, 2019)
=========================
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

Pre-1.0.0 releases
==================

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
