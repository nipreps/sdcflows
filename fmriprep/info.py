# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

__author__ = 'The CRN developers'
__copyright__ = 'Copyright 2018, Center for Reproducible Neuroscience, Stanford University'
__credits__ = ['Craig Moodie', 'Ross Blair', 'Oscar Esteban', 'Chris Gorgolewski',
               'Shoshana Berleant', 'Christopher J. Markiewicz', 'Russell A. Poldrack']
__license__ = '3-clause BSD'
__maintainer__ = 'Ross Blair'
__email__ = 'crn.poldracklab@gmail.com'
__status__ = 'Prototype'
__url__ = 'https://github.com/poldracklab/fmriprep'
__packagename__ = 'fmriprep'
__description__ = ("fMRIprep is a functional magnetic resonance image pre-processing pipeline "
                   "that is designed to provide an easily accessible, state-of-the-art interface "
                   "that is robust to differences in scan acquisition protocols and that requires "
                   "minimal user input, while providing easily interpretable and comprehensive "
                   "error and output reporting.")
__longdesc__ = """\
Preprocessing of functional MRI (fMRI) involves numerous steps to clean and standardize
data before statistical analysis. Generally, researchers create ad hoc preprocessing
workflows for each new dataset, building upon a large inventory of tools available for
each step. The complexity of these workflows has snowballed with rapid advances in MR data
acquisition and image processing techniques. We introduce fMRIPrep, an analysis-agnostic
tool that addresses the challenge of robust and reproducible preprocessing for task-based
and resting fMRI data. FMRIPrep automatically adapts a best-in-breed workflow to the
idiosyncrasies of virtually any dataset, ensuring high-quality preprocessing with no
manual intervention. By introducing visual assessment checkpoints into an iterative
integration framework for software-testing, we show that fMRIPrep robustly produces
high-quality results on a diverse fMRI data collection comprising participants from
54 different studies in the OpenfMRI repository. We review the distinctive features of
fMRIPrep in a qualitative comparison to other preprocessing workflows. FMRIPrep achieves
higher spatial accuracy as it introduces less uncontrolled spatial smoothness than commonly
used preprocessing tools. FMRIPrep has the potential to transform fMRI research by equipping
neuroscientists with a high-quality, robust, easy-to-use and transparent preprocessing workflow
which can help ensure the validity of inference and the interpretability of their results.

[Pre-print https://doi.org/10.1101/306951]"""

DOWNLOAD_URL = (
    'https://github.com/poldracklab/{name}/archive/{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))


SETUP_REQUIRES = [
    'setuptools>=18.0',
    'numpy',
    'cython',
]

REQUIRES = [
    'numpy',
    'lockfile',
    'future',
    'scikit-learn',
    'matplotlib>=2.2.0',
    'nilearn',
    'sklearn',
    'nibabel>=2.2.1',
    'pandas',
    'grabbit',
    'pybids>=0.6.3',
    'nitime',
    'nipype>=1.1.1',
    'niworkflows>=0.4.2',
    'statsmodels',
    'seaborn',
    'indexed_gzip>=0.8.2',
    'scikit-image',
    'versioneer',
]

LINKS_REQUIRES = [
]

TESTS_REQUIRES = [
    "mock",
    "codecov",
    "pytest",
]

EXTRA_REQUIRES = {
    'doc': [
        'sphinx>=1.5.3',
        'sphinx_rtd_theme',
        'sphinx-argparse',
        'pydotplus',
        'pydot>=1.2.3',
        'packaging',
        'nbsphinx',
    ],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit'],
    'datalad': ['datalad'],
    'resmon': ['psutil>=5.4.0'],
    'sentry': ['raven'],
}
EXTRA_REQUIRES['docs'] = EXTRA_REQUIRES['doc']

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = list(set([
    v for deps in EXTRA_REQUIRES.values() for v in deps]))

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
]
