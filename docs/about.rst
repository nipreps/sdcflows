About
-----

``fmriprep`` is a functional magnetic resonance image preprecessing pipeline
that is designed to provide an easily accessible, state-of-the-art interface
that is robust to differences in scan acquisition protocols and that requires
minimal user input, while providing easily interpretable and comprehensive
error and output reporting. This open-source neuroimaging data processing tool
is being developed as a part of the MRI image analysis and reproducibility
platform offered by the CRN. This pipeline is heavily influenced by the Human
Connectome Project analysis pipelines
(https://github.com/Washington-University/Pipelines) and, as such, the
backbone of this pipeline is a python reimplementation of the HCP
GenericfMRIVolumeProcessingPipeline.sh script. However, a major difference is
that this pipeline is executed using a Nipype workflow framework. This allows
for each call to a software module or binary to be controlled within the
workflows, which removes the need for manual curation at every stage, while
still providing all the output and error information that would be necessary
for debugging and interpretation purposes. The fmriprep pipeline primarily
utilizes FSL tools, but also utilizes ANTs tools at several stages such as
skull stripping and template registration. This pipeline was designed to
provide the best software implementation for each state of preprocessing, and
will be updated as newer and better neuroimaging software become available.

This tool allows you to easily do the following:

- Take fMRI data from raw to full preprocessed form.
- Implement tools from different software packages.
- Achieve optimal data processing quality by using the best tools available.
- Generate preprocessing quality reports, with which the user can easily
identify outliers.
- Receive verbose output concerning the stage of pre-processing for each
subject, including meaningful errors.
- Automate and parallelize processing steps, which provides a significant
speed-up from typical linear, manual processing.

More information and documentation can be found here:

https://preprocessing-workflow.readthedocs.io/
