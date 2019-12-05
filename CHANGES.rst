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
