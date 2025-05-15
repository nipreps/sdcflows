2.13.0 (May 15, 2025)
=====================
Feature release in the 2.13.x series.

This release addresses some longstanding issues with the SyN-SDC workflow,
improving the registration quality in adult humans by utilizing a spatial prior,
as well as allowing Laplacians to be up- or down-weighted in the cost function,
making it more usable across species.

Additionally, this release allows for the use of ``EstimatedTotalReadoutTime`` or
``EstimatedEchoSpacing``, or a manually provided fallback ``TotalReadoutTime`` value,
permitting the use of SDCFlows on datasets that do not have reliable timing information
without introducing incorrect metadata into the datasets.

* fix(syn): Re-enable priors respecting ``sd_priors`` argument (#488)
* feat: Add workflow arguments for metadata estimates and fallback TRT (#479)
* feat(syn): Update totalFieldVarianceInVoxel space based on voxel resolution (#487)
* feat(syn): Allow changing laplacians weights in SyN registration metric (#484)
* test(syn): Add a test to exercise SyN workflow creation and check parameters (#486)


2.12.0 (March 21, 2025)
=======================
Feature release in the 2.12.x series.

This release migrates from the deprecated ``niworkflows.reporting``
module to the ``nireports`` package.

* FIX: AttributeError for _ApplyCoeffsFieldInputSpec (#481)
* ENH: Allow running SyN SDC without using prior (#480)
* ENH: Allow estimated and fallback TotalReadoutTime (#477)
* RF: Transition from niworkflows reporting interfaces (#473)
* DOC: Fix broken link [skip ci] (#482)
* MNT: Add `defaults` to `conda` channels in `build-test-publish` GHA (#474)
* MNT: Update `niworkflows` version to 1.11.0 (#478)


2.11.0 (December 18, 2024)
==========================
Feature release in the 2.11.x series.

This release supports numpy 2 and Python 3.13.

* FIX: Normalize BIDS-URIs to subject-relative (#458)
* FIX: Only fit high-frequency spline for SyN-SDC (#457)
* ENH: Allow Jacobian correction to be toggled on/off (#462)
* ENH: Dilate fmap and bold masks during coregistration (#463)
* TEST: Clear registry consistently to avoid order dependency (#464)
* DOC: Plot correct workflow in init_syn_preprocessing_wf docstring (#460)
* CI: Drop CircleCI, using GHA only (#459)


2.10.0 (July 04, 2024)
======================
Feature release in the 2.10.x series.

The main change is a bugfix when fitting multi-level B-Splines to some
noisy fieldmaps.
The multi-level fitting, while being theoretically nice, does not really
bring anything immediately as we are not generally inverting the distortion.
In this release, the default fitting has been change to single-level B-Splines,
with a spacing similar to TOPUP's defaults.

With thanks to Sunjae Shim (@sjshim) for sharing Spiral-echo fieldmaps that
were failing prior the patch in #453.

CHANGES
-------

* FIX: Revision of the B-Spline fitting code (#453)
* FIX: Building Docker image on ARM64 devices (#449)
* ENH: Improve plots in jupyter notebook (#452)
* DOC: Fix ``ValueError`` in notebook's output. (#450)


2.9.0 (June 13, 2024)
=====================
Feature release in the 2.9.x series.

The main change is that B0FieldIdentifiers with special characters
are now sanitized and exposed through a ``FieldmapEstimator.sanitized_id``
attribute.

Workflow names and input/output connections will use ``sanitized_id``,
to ensure compatibility with filenames and workflow graph generation.
Internal lookup tables will continue to use the the unsanitized ``bids_id``.

* FIX: Update suffix only when finding related fieldmap files (#436)
* FIX: Remove unused ANTs parameter that was removed in 2.4.1 (#431)
* RF: Add sanitized_id field to FieldmapEstimation (#444)
* DOC: Un-mock the already-imported numpy (#440)
* CI: Bump actions/cache from 3 to 4 (#429)

2.8.1 (January 22, 2024)
========================
Bug-fix release in the 2.8.x series.

Fixes doc builds and CLI support for fieldmapless workflows.
Introduces support for BIDS-URIs.

* FIX: Enable fieldmapless by default in CLI (#426)
* FIX: Pandoc requires Texlive to render LaTeX in notebook (#427)
* FIX: New test ``test_wrangler_URIs`` had the wrong oracle (#425)
* ENH: Resolve BIDS-URIs (#349)
* TEST: Use less confusing function name for testing CLI with --dry-run (#424)
* MNT: Bump actions/download-artifact from 3 to 4 (#418)
* MNT: Bump actions/upload-artifact from 3 to 4 (#417)
* CI: Move PR doc build into main doc build, add texlive to build dependencies (#428)

2.8.0 (January 10, 2024)
========================
New feature release in the 2.8.x series.

This release fixes a bug in converting SyN-SDC displacements to fieldmaps,
resulting in exaggerated corrections. As this makes changes to the structure
of a workflow and the expected inputs of a workflow node, this release is
considered a minor release.

* FIX: Derive field from SyN displacements using EPI affine (#421)
* FIX: Change ``os.basename`` to ``os.path.basename`` (#419)
* DOC: Add @smeisler to contributors (#420)

2.7.0 (December 18, 2023)
=========================
New feature release in the 2.7.0 series.

This release includes an updated CLI, which allows ``sdcflows`` to be
run as a BIDS App. To achieve the previous behavior of ``sdcflows-find-estimators``,
use the ``-n`` flag.

Addtional bug fixes and enhancements are included.

* FIX: Drop header before resampling image to avoid unsafe cast (#415)
* FIX: Wrangler now ignores ``part-phase`` EPI files (#407)
* ENH: Standalone CLI now estimates fieldmaps (#408)
* ENH: Add support for ASL data (#411)
* ENH: Enable rendering of the Jupyter notebooks (#409)
* MNT: Migrate to PEP517/518 packaging (#410)
* CI: bump actions/setup-python from 4 to 5 (#412)
* CI: bump conda-incubator/setup-miniconda from 2 to 3 (#406)

2.6.0 (November 10, 2023)
=========================
New feature release in the 2.6.0 series.

This release resolves a number of issues with fieldmaps inducing distortions
during correction. Phase difference and direct fieldmaps are now masked correctly,
preventing the overestimation of distortions outside the brain. Additionally,
we now implement Jacobian weighting during unwarping, which corrects for compression
and expansion effects on signal intensity.

* FIX: Mask fieldmap before fitting spline field (#396)
* FIX: Interpolate to floating point values (#394)
* FIX: Refactoring the ``B0FieldTransform`` implementation (#346)
* FIX: Nipype workflows like to be passed absolute paths (phasediff fieldmap) (#374)
* ENH: Implement Jacobian weighting during unwarp (#391)
* ENH: Output target2fmap_xfm from coeff2epi_wf (#381)
* ENH: Add data loader to sdcflows.data, drop pkg_resources (#379)
* RF: Use scipy.interpolate.BSpline to construct spline basis (#393)
* DOC: Use latest sphinx to fix bad sphinx/furo interaction (#390)
* DOC: Fix missing dependency when merging new data loader (#380)
* MNT: Update emprical values in test to allow transition to new scipy's BSpline (#387)
* MNT: Add pre-commit config (#375)
* MNT: Add a seed to random generator of coefficients (#368)

2.5.2 (November 09, 2023)
=========================
Bug-fix release in the 2.5.x series.

This release includes a fix for phasediff/direct fieldmaps that were previously
producing distortions outside the brain due to an incorrect masking of the fieldmap.

* FIX: Mask fieldmap before fitting spline field [backport gh-396] (#398)
* DOC: Fix doc build for 2.5.x branch (#399)
* MAINT: Make call to scipy.stats.mode compatible with scipy 1.11.0 (#371)

2.5.1 (June 08, 2023)
=====================
Bug-fix release in the 2.5.x series.

* FIX: Use ``lsqr`` solver for spline fit, rerun on extreme values (#366)
* FIX: Ensure metadata is not present in entity query (#367)
* RF/FIX: Prioritize sbref and shortest echo for SyN-SDC (#364)

2.5.0 (June 01, 2023)
=====================
New feature release in the 2.5.x series.

This release includes a number of changes to default behaviors.
SyN-SDC will be performed per-BOLD/DWI image, unless specified otherwise with
``B0FieldIdentifier``\s, and may now be specified with T2w images as anatomical
references as well.
Additionally, PEPolar fieldmaps will only be grouped if they share ``IntendedFor``
metadata.

Finally, as a small UX improvement, if magnitude1/magnitude2 images have differing
affines but are in register, we will now copy the header rather than requiring the
user to update the header themselves.

* FIX: Ensure IntendedFor metadata is a subject-relative path (#360)
* ENH: Split SyN fieldmap estimates per-EPI (#312)
* ENH: Allow non-T1w anatomical estimators (#358)
* ENH: Function to calculate reference grids aligned with the coefficients (#355)
* ENH: Check registration of magnitude1/magnitude2 images and update headers (#356)
* RF: Split PEPolar fieldmaps by intent, if available (#342)
* CI: Use supported codecov uploaders (#348)

2.4.3 (April 24, 2023)
======================
Bug-fix release in the 2.4.x series.

This fix resolves an inconsistency of treatment of phase-difference and
scanner-calculated fieldmaps, relative to PEPolar and SyN. Fieldmaps in
orientations other than RAS were impacted.

* FIX: Reorient fieldmaps to RAS before estimating B-splines (#354)

2.4.2 (April 20, 2023)
======================
Bug-fix release in the 2.4.x series.

Same fixes as 2.4.1, but this time for phase-difference and direct fieldmaps
we missed last time.

* FIX: Capture and report partial fieldmaps (#351)

2.4.1 (March 20, 2023)
======================
Bug-fix release in the 2.4.x series.

This release provides improved tolerance (and debugging output)
for incomplete fieldmap inputs.

* FIX: Log incomplete fieldmaps, rather than erroring (#341)
* ENH: Consistently log failures to form fieldmaps (#343)

2.4.0 (March 10, 2023)
======================
New feature release in the 2.4.x series.

This release supports fMRIPrep 23.0.x and Nibabies 23.1.x.

* FIX: Reorient phase-encoding directions along with fieldmaps when preparing inputs to TOPUP (#339)
* FIX: Correct overly-sensitive obliqueness check (#335)

2.3.0 (March 01, 2023)
======================
New feature release in the 2.3.x series.

This release supports fMRIPrep 23.0.x and Nibabies 23.0.x.

* ENH: Calculate fieldwarps in reference space in unwarp_wf (#334)
* TEST: Squeeze image before passing to SimpleBeforeAfter (#337)
* MAINT: Rotate CircleCI secrets and setup up org-level context (#329)
* CI: Run unit tests on Python 3.10 (#326)
* CI: Switch to miniconda setup, install fsl tools through conda (#322)

2.2.2 (January 04, 2023)
========================
Patch release in the 2.2.x series.

This release resolves a bug affecting some oblique datasets.

* RF: Generate the B-spline design matrix directly for efficiency (#324)
* DOC: Add a notebook about susceptibility distortions (#285)


2.2.1 (December 12, 2022)
=========================
Patch release in the 2.2.x series.

This release enables dynamic estimation of memory and CPU needs for a
particularly resource-intensive node.

* ENH: Dynamically choose number of resampling threads to adapt to memory constraints (#321)


2.2.0 (December 09, 2022)
=========================
New feature release in the 2.2.x series.

This series supports fMRIPrep 22.1.x and Nibabies 22.2.x.

This release includes fixes for a number of SDC use cases.

With thanks to Basile Pinsard for adding support for fieldmaps
that contribute to multiple ``B0FieldIdentifier``\s.

.. attention::

    *SDCFlows* drops Python 3.7 starting with 2.2.x series.

* FIX: Collate fieldmap coefficients into list of lists (#317)
* FIX: Pad BSpline design matrix (#319)
* FIX: Calculate bspline grids separately from colocation matrices (#308)
* FIX: Support scipy 1.8 (#311)
* FIX: Pacify deprecation warning from scipy.stats (#309)
* FIX: Do not reorient distorted image in apply (#303)
* FIX: Do not create a dense matrix along the way (#299)
* FIX: Ensure ``replace()`` calls only alter the file basename (#293)
* FIX: Update tests after merge of #287 (#288)
* FIX: Revise debug/sloppy operations of the ``coeff2epi`` workflow (#287)
* FIX: Revise the TOPUP workflow and related utilities (#278)
* ENH: Default to 4mm re-zoom for b-spline approximation (#314)
* ENH: Drop n_procs tag from BSplineApprox (#315)
* ENH: Find B0FieldIdentifiers when one image contributes to multiple (#298)
* ENH: Allow bids filtering during ``get()`` calls. (#292)
* ENH: Evaluate B-Splines using scipy (#304)
* ENH: Integrate downsampling in ``BSplineApprox`` when the input is high-res (#301)
* ENH: Make wrangler more verbose (#284)
* ENH: Add CLI to detect usable estimators within a BIDS dataset (#257)
* ENH: Calculate robust average of EPI inputs to TOPUP workflow (#280)
* MAINT: Housekeeping and more verbose debugging outputs (#302)
* MAINT: Simplify build tests on GH Actions to latest standards (#282)
* MAINT: Keep CircleCI settings up to date (#281)
* MAINT: Unavilable data from OSF remote (datalad) for CircleCI tests. (#277)
* MAINT: Remove unused argument from ``topup`` related interface (#276)
* CI: Update concurrency, permissions and actions (#313)
* CI: Roll unittests runner back to Ubuntu 20.04 (#310)
* CI: Ensure builds are triggered weekly (#270)

2.1.1 (August 29, 2022)
=======================
Patch release in the 2.1.x series. This release incorporates the fix in 2.0.13 in
the 2.1.x series.

* FIX: Relax tolerance for different affines when concatenating blips (#265)

2.1.0 (May 26, 2022)
====================
A new minor release to support the newest niworkflows minor series.

  * ENH: Add optional session distinction to wrangler (#261)
  * FIX: Align centers of mass, rather than origins (#253)
  * MAINT: Loosen installation restrictions (#269)

2.0.13 (April 08, 2022)
=======================
Patch release in the 2.0.x series. This release resolves an issue in fMRIPrep 21.0.x.

* FIX: Relax tolerance for different affines when concatenating blips (#265)

2.0.12 (February 08, 2022)
==========================
Patch release in the 2.0.x series. This allows compatibility with the next minor release of ``niworkflows``.

* MAINT: Allow compatibility with new niworkflows minor (#262)
* DOC: Update scipy intersphinx url (#263)

2.0.11 (January 22, 2022)
==========================
Patch release in the 2.0.x series.

* FIX: Create one fieldmap estimator per EPI-IntendedFor pair (#258)
* DOCKER: Build with FSL 6 (#254)

2.0.10 (December 13, 2021)
==========================
Patch release in the 2.0.x series.

* FIX: Update boilerplate ordering directives (#229)
* FIX: ishandling of ``topup`` coefficients with higher resolution EPIs (#251)

2.0.9 (November 16, 2021)
=========================
A patch release improving documentation and implementing ``B0Field*`` BIDS metadata.

* DOC: Bring implementation details to the foreground of documentation (#248)
* FIX: Implement ``B0FieldIdentifier`` / ``B0FieldSource`` (#247)

2.0.8 (October 15, 2021)
========================
A patch release with a deep revision of the new implementation of the fieldmap-less "*SDC-SyN*" toward integration with *fMRIPrep*.

* FIX: *SDC-SyN* ("fieldmap-less") overhaul (#239)
* DOC: Self-hosted & multiversion documentation overhaul (#243)
* MAINT: Standardization of containers across *NiPreps* (#240)

2.0.7 (September 30, 2021)
==========================
A patch release with important bugfixes discovered during the integration with *fMRIPrep*.

* FIX: Generation of *RAS* displacements fields from *VSM*\ s (#237)
* FIX: Use subject root to resolve ``IntendedFor`` paths (#228)
* ENH: Improve support of 4D in ``sdcflows.interfaces.bspline.ApplyCoeffsField`` (#234)
* MAINT: Update node and ``gh-pages``, push docs without history (#230)

2.0.6 (September 1, 2021)
=========================
A patch release to address a problem with TOPUP and an odd number of slices.

* ENH: Add slice padding to TOPUP (#217)

2.0.5 (August 24, 2021)
=======================
A bugfix release, adds the fieldwarp as an output to the unwarping workflow.

* FIX: The calculated displacements field not exposed by unwarp workflow (#224)
* MAINT: Use keys.openpgp.org over sks-keyservers (#223)

2.0.4 (May 18, 2021)
====================
A hotfix release including some enhancements that should have been released within
the previous 2.0.3 release.

* ENH: Fine-tune the registration parameters in ``coeff2epi`` workflow (#215)
* ENH: Finalize upstreaming to *NiWorkflows* of ``IntensityClip`` (#216)
* ENH: Use new ``RobustAverage`` interface to merge corrected blips (#214)
* DOC: Insert copyright notice in header comments as per Apache-2.0 (#212)

2.0.3 (May 14, 2021)
====================
A patch release including some improvements to the PEPOLAR/TOPUP implementation,
along with corresponding updates to the CI tests.

* ENH: Uniformize the grid&affine across EPI "blips" before TOPUP (#197)
* MAINT: Fix PEPOLAR workflow test with HCP data (#210)
* MAINT: Update tests after changes in ds001771's structure (#209)

2.0.2 (May 11, 2021)
====================
A patch release including hot-fixes and some relevant improvements inteded for the reliability
of the new API.
The most relevant advance is the new :math:`B_0` fieldmap unwarping object which is compatible
with *NiTranforms* and evades the problem of fiddling with the target image's x-forms.

* FIX: Make sure the VSM is not modified when accessing it (#207)
* FIX: Normalize phase-encoding polarity of coefficients after TOPUP (#202)
* FIX: Revise generation of the displacements field from coefficients (#199)
* FIX: Inconsistency left after renaming inputs to SDC-SyN (removing "BOLD") (#182)
* FIX: Correctly interpolate the BIDS root when datasets have sessions (#180)
* ENH: :math:`B_0` fieldmap unwarping object (#204)
* ENH: Add estimation method description to outputs (#191)
* ENH: Ensure a function node is covered with unit tests (#188)
* ENH: Add a preprocessing pipeline for SDC-SyN (#184)
* ENH: [rodents] Add input to override default B-Spline distances in INU correction with N4 (#178)
* ENH: Adopt new brain extraction algorithm in magnitude preparation workflow (#176)
* DOC: Fix typos as per codespell (#205)
* MAINT: Double-check conversion from TOPUP to standardized fieldmaps (#200)
* MAINT: Divide ambiguous debug parameter into smaller, more focused parameters (#190)
* MAINT: Adapt to GitHub actions' upgrade to Ubuntu 20.04 (#185)

2.0.1 (March 05, 2021)
======================
A patch release including some bugfixes and minimal improvements over the previous
major release.

* FIX: Inconsistency left after renaming inputs to SDC-SyN (removing "BOLD") (#182)
* FIX: Correctly interpolate the BIDS root when datasets have sessions (#180)
* ENH: Add a preprocessing pipeline for SDC-SyN (#184)
* ENH: [rodents] Add input to override default B-Spline distances in INU correction with N4 (#178)
* ENH: Adopt new brain extraction algorithm in magnitude preparation workflow (#176)
* MAINT: Adapt to GitHub actions' upgrade to Ubuntu 20.04 (#185)

2.0.0 (January 25, 2021)
========================
The *SDCFlows* 2.0.x series are released after a comprehensive overhaul of the software's API.
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

.. attention::

    *SDCFlows* drops Python 3.6 starting with 1.4.x series.

Some of the most prominent pull-requests conducive to this release are:

* FIX: Fast & accurate brain extraction of magnitude images without FSL BET (#174)
* FIX: svgutils 0.3.2 breaks our reportlets (#175)
* FIX: Misconfigured test of unwarping workflow (#170)
* FIX: Cleanup annoying isolated dots in reportlets + new tests (#168)
* FIX: Make images "plumb" before running ANTs-SyN (and roll-back afterwards) (#165)
* FIX: Convert SEI fieldmaps given in rad/s into Hz (#127)
* FIX: Limit ``3dQwarp`` to maximum 4 CPUs for stability reasons (#128)
* ENH: Adopt new brain extraction algorithm in magnitude preparation workflow (#176)
* ENH: Add "*fieldmap-less*" estimations to default heuristics (#166)
* ENH: Add one test for the SDC-SyN workflow (#164)
* ENH: Generate a simple mask after correction (#161)
* ENH: Increase unit-tests coverage of ``sdcflows.fieldmaps`` (#159)
* ENH: Optimize tensor-product B-Spline kernel evaluation (#157)
* ENH: Add a memory check to dynamically limit interpolation blocksize (#156)
* ENH: Massage TOPUP's fieldcoeff files to be compatible with ours (#154)
* ENH: Add a simplistic EPI masking algorithm (#152)
* ENH: Utility that returns the ``B0FieldSource`` of a given potential EPI target (#151)
* ENH: Write ``fmapid-`` entity in Derivatives (#150)
* ENH: Multiplex fieldmap estimation outputs into a single ``outputnode`` (#149)
* ENH: Putting the new API together on a base workflow (#143)
* ENH: Autogenerate ``B0FieldIdentifiers`` from ``IntendedFor`` (#142)
* ENH: Finalizing the API overhaul (#132)
* ENH: Keep a registry of already-used identifiers (and auto-generate new) (#130)
* ENH: New objects for better representation of fieldmap estimation (#114)
* ENH: Show FieldmapReportlet oriented aligned with cardinal axes (#120)
* ENH: New estimation API (featuring a TOPUP implementation!) (#115)
* DOC: Minor improvements to the literate workflows descriptions. (#162)
* DOC: Fix typo in docstring (#155)
* DOC: Enable NiPype's sphinx-extension to better render Interfaces (#131)
* MAINT: Docker - Update base Ubuntu image & ANTs, makefile (#173)
* MAINT: Retouch several tests and improve ANTs version handling of SyN workflow (#172)
* MAINT: Drop Python 3.6 (#160)
* MAINT: Enable Git-archive protocol with setuptools-scm-archive (#153)
* MAINT: Migrate TravisCI -> GH Actions (completion) (#138)
* MAINT: Migrate TravisCI -> GH Actions (#137)
* MAINT: Minimal unit test and refactor of pepolar workflow node (#133)
* MAINT: Collect code coverage from tests on Circle (#122)
* MAINT: Test minimum dependencies with TravisCI (#96)
* MAINT: Add FLIRT config files to prepare for TOPUP integration (#116)

A complete list of issues addressed by the release is found `in the GitHub repo
<https://github.com/nipreps/sdcflows/milestone/2?closed=1>`__.

.. admonition:: Author list for papers based on *SDCFlows* 2.0.x series

    As described in the `Contributor Guidelines
    <https://www.nipreps.org/community/CONTRIBUTING/#recognizing-contributions>`__,
    anyone listed as developer or contributor may write and submit manuscripts
    about *SDCFlows*.
    To do so, please move the author(s) name(s) to the front of the following list:

    Markiewicz, Christopher J. \ :sup:`1`\ ; Goncalves, Mathias \ :sup:`1`\ ; MacNicol, Eilidh \ :sup:`2`\ ; Adebimpe, Azeez \ :sup:`3`\ ; Blair, Ross W. \ :sup:`1`\ ; Cieslak, Matthew \ :sup:`3`\ ; Naveau, Mikaël \ :sup:`4`\ ; Sitek, Kevin R. \ :sup:`5`\ ; Sneve, Markus H. \ :sup:`6`\ ; Gorgolewski, Krzysztof J. \ :sup:`1`\ ; Satterthwaite, Theodore D. \ :sup:`3`\ ; Poldrack, Russell A. \ :sup:`1`\ ; Esteban, Oscar \ :sup:`7`\ .

    Affiliations:

    1. Department of Psychology, Stanford University
    2. Department of Neuroimaging, King's College London
    3. Perelman School of Medicine, University of Pennsylvania, PA, USA
    4. Cyceron, UMS 3408 (CNRS - UCBN), France
    5. Speech & Hearing Bioscience & Technology Program, Harvard University
    6. Center for Lifespan Changes in Brain and Cognition, University of Oslo
    7. Dept. of Radiology, Lausanne University Hospital, University of Lausanne

1.3.x series
============

1.3.5 (February 14, 2024)
-------------------------
Bug-fix release in 1.3.x series.

* FIX: Remove unused ANTs parameter that was removed in 2.4.1 (#431)

1.3.4 (July 07, 2023)
---------------------
Bug-fix release in 1.3.x series.

* FIX: Limit ``3dQwarp`` to maximum 4 CPUs for stability reasons (#128)
* MAINT: Make call to scipy.stats.mode compatible with scipy 1.11.0 (#371)
* CI: Update docker/machine images for 1.3.x branch (#327)

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
