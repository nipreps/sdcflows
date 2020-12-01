SDCFlows
--------
.. image:: https://img.shields.io/pypi/v/sdcflows.svg
  :target: https://pypi.python.org/pypi/sdcflows/
  :alt: Latest Version

.. image:: https://codecov.io/gh/nipreps/sdcflows/branch/master/graph/badge.svg?token=V2CS5adHYk
  :target: https://codecov.io/gh/nipreps/sdcflows

.. image:: https://circleci.com/gh/nipreps/sdcflows.svg?style=svg
    :target: https://circleci.com/gh/nipreps/sdcflows

.. image:: https://github.com/nipreps/sdcflows/workflows/Deps%20&%20CI/badge.svg
    :target: https://github.com/nipreps/sdcflows/actions

SDCFlows (*Susceptibility Distortion Correction workFlows*) is a Python library of 
*NiPype*-based workflows to preprocess *B0* mapping data, estimate the corresponding
fieldmap and finally correct for susceptibility distortions.
Susceptibility-derived distortions are typically displayed by images acquired with EPI
(echo-planar imaging) MR schemes.

The library is designed to provide an easily accessible, state-of-the-art interface that is
robust to differences in scan acquisition protocols and that requires minimal user input.

This open-source neuroimaging data processing tool is being developed as a part of
the MRI image analysis and reproducibility platform offered by
`NiPreps <https://www.nipreps.org>`__.
