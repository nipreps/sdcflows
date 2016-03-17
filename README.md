-

==========================================================
FMRIPREP: A Robust Preprocessing Pipeline for fMRI Data
==========================================================

This pipeline is developed by the Poldrack lab at Stanford University (https://poldracklab.stanford.edu/) for use at the Center for Reproducible Neuroscience (http://reproducibility.stanford.edu/), as well as for open-source software distribution.

About
=====

``frmiprep`` is a functional magnetic resonance image preprecessing pipeline that is designed to provide an easily accessible, state-of-the-art interface that is robust to differences in scan acquisition protocols and that requires minimal user input, while providing easily interpretable and comprehensive error and output reporting. This open-source neuroimaging data processing tool is being developed as a part of the MRI image analysis and reproducibility platform offered by the CRN. This pipeline is heavily influenced by the Human Connectome Project analysis pipelines (https://github.com/Washington-University/Pipelines) and, as such, the backbone of this pipeline is a python reimplementation of the HCP GenericfMRIVolumeProcessingPipeline.sh script. However, a major difference is that this pipeline is executed using a Nipype workflow framework. This allows for each call to a software module or binary to be controlled within the workflows, which removes the need for manual curation at every stage, while still providing all the output and error information that would be necessary for debugging and interpretation purposes. The fmriprep pipeline primarily utilizes FSL tools, but also utilizes ANTs tools at several stages such as skull stripping and template registration. This pipeline was designed to provide the best software implementation for each state of preprocessing, and will be updated as newer and better neuroimaging software become available.

This tool allows you to easily do the following:

* Take fMRI data from raw to full preprocessed form.
* Implement tools from different software packages.
* Achieve optimal data processing quality by using the best tools available.
* Generate preprocessing quality reports, with which the user can easily identify outliers.
* Receive verbose output concerning the stage of pre-processing for each subject, including meaningful errors.
* Automate and parallelize processing steps, which provides a significant speed-up from typical linear, manual processing.

More information and documentation can be found here: 

https://github.com/poldracklab/preprocessing-workflow


Dependencies
============

#. `FSL <http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>`_
#. `ANTs <http://stnava.github.io/ANTs/>`_
#. `Nipype <http://nipy.org/nipype/>`_
#. `NumPy <http://www.numpy.org/>`_
#. `Matplotlib <http://matplotlib.org/>`_
#. `PyYAML <http://pyyaml.org/>`_
#. `MRIQC <https://github.com/poldracklab/mriqc>`_
#. `NiBabel <http://nipy.org/nibabel/>`_
#. `PyLab <http://scipy.github.io/old-wiki/pages/PyLab>`_


All requirements are listed in ``requirements.txt`` file.


Installation
============




Usage
=====



Reporting Issues
================

All bugs, concerns and enhancement requests for this software can be submitted here:

https://github.com/poldracklab/preprocessing-workflow/issues.


Authors
=======

Craig A. Moodie, Krzysztof J. Gorgolewski, Oscar Esteban, Ross Blair.
Poldrack Lab, Psychology Department, Stanford University.
Center for Reproducible Neuroscience, Stanford University.

License
=======

Copyright (c) 2016, 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of preprocessing-workflow nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=======
The fMRI preprocessing workflow
===============================

-
