#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
TopUp helpers
~~~~~~~~~~~~~


"""
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path as op
import numpy as np
import nibabel as nb
from nilearn.image import mean_img, concat_imgs
from nilearn.masking import (compute_epi_mask,
                             apply_mask)
from nipype import logging
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces import fsl
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec,
    File, InputMultiPath, traits, OutputMultiPath
)
from fmriprep.utils.misc import genfname

from .images import reorient
from .bids import get_metadata_for_nifti

LOGGER = logging.getLogger('interface')
PEPOLAR_MODALITIES = ['epi', 'sbref']

class TopupInputsInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True),
                              desc='input file for topup field estimation')
    to_ras = traits.Bool(True, usedefault=True,
                         desc='reorient all input images to RAS')
    mask_inputs = traits.Bool(True, usedefault=True,
                              desc='do mask of inputs')
    nthreads = traits.Int(-1, usedefault=True,
                          desc='number of threads to use within ANTs registrations')

class TopupInputsOutputSpec(TraitedSpec):
    out_blips = traits.List(traits.Tuple(traits.Float, traits.Float),
                            desc='List of encoding files')
    out_file = File(exists=True, desc='combined input file as TopUp wants them')
    out_filelist = OutputMultiPath(
        File(exists=True), desc='list of output files as ApplyTOPUP wants them')
    out_encfile = File(exists=True, desc='encoding file corresponding to datain')


class TopupInputs(BaseInterface):

    """
    This interface generates the input files required by FSL topup:

      * A 4D file with all input images
      * The topup-encoding parameters file corresponding to the 4D file.


    """
    input_spec = TopupInputsInputSpec
    output_spec = TopupInputsOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(TopupInputs, self).__init__(**inputs)

    def _list_outputs(self):
        return self._results

    def _run_interface(self, runtime):

        in_files = [fname for fname in self.inputs.in_files
                    if 'epi' in fname]
        in_files += [fname for fname in self.inputs.in_files
                     if 'sbref' in fname]

        LOGGER.info('TopUp inputs: %s', ', '.join(in_files))

        nthreads = self.inputs.nthreads
        if nthreads < 1:
            from multiprocessing import cpu_count
            nthreads = cpu_count()


        # Check input files.
        if not isinstance(in_files, (list, tuple)):
            raise RuntimeError('in_files should be a list of files')
        if len(in_files) < 2:
            raise RuntimeError('in_files should be a list of 2 or more input files')

        # Check metadata of inputs and retrieve the pe dir and ro time.
        prep_in_files = []
        out_encodings = []

        for fname in in_files:
            # Get metadata (pe dir and echo time)
            pe_dir, ectime = get_pe_params(fname)

            # check number of images in dataset
            nii = nb.squeeze_image(nb.load(fname))
            ntsteps = nii.shape[-1] if len(nii.shape) == 4 else 1

            seq_names = [fname]
            if ntsteps > 1:
                # Expand 4D files
                nii_list = nb.four_to_three(nii)
                seq_names = []
                for i, frame in enumerate(nii_list):
                    newfname = genfname(fname, suffix='seq%02d' % i)
                    seq_names.append(newfname)
                    frame.to_filename(newfname)

            # to RAS
            if self.inputs.to_ras:
                seq_names = [reorient(sname) for sname in seq_names]

            out_encodings += [(pe_dir, ectime)] * ntsteps
            prep_in_files += seq_names

        if len(out_encodings) != len(prep_in_files):
            raise RuntimeError('Length of encodings and files should match')

        # Find unique sorted
        blips = []
        for el in out_encodings:
            try:
                blips.index(el)
            except ValueError:
                blips.append(el)

        LOGGER.info('Unique blips found: %s', ', '.join(str(b) for b in blips))

        if len(blips) < 2:
            raise RuntimeError(
                '"PEpolar" methods require for files to be encoded at least'
                ' with two different phase-encoding axes/directions.')

        pe_files = []
        encoding = np.zeros((len(blips), 4))
        for i, blip in enumerate(blips):
            blip_files = [fname for enc, fname in zip(out_encodings, prep_in_files)
                          if enc == blip]
            LOGGER.info('Running motion correction on files: %s', ', '.join(blip_files))
            pe_files.append(_coregistration(blip_files, nthreads=nthreads)[0])

            encoding[i, int(abs(blip[0]))] = 1.0 if blip[0] > 0 else -1.0
            encoding[i, 3] = blip[1]

        self._results['out_blips'] = blips
        self._results['out_encfile'] = genfname(in_files[0], suffix='encfile', ext='txt')
        np.savetxt(self._results['out_encfile'], encoding,
                   fmt=['%0.1f'] * 3 + ['%0.6f'])


        out_files = [pe_files[0]]
        for i, moving in enumerate(pe_files[1:]):
            LOGGER.info('Running coregistration of %s to reference %s', moving, pe_files[0])
            out_files.append(_run_registration(pe_files[0], moving,
                             prefix=op.basename(genfname(moving, suffix='reg%02d' % i))))

        out_file = genfname(out_files[0], suffix='datain')
        concat_imgs(out_files).to_filename(out_file)
        self._results['out_file'] = out_file
        self._results['out_filelist'] = out_files

        return runtime


def get_pe_params(in_file):
    """
    Checks on the BIDS metadata associated with the file and
    extracts the two parameters of interest: PE direction and
    RO time.
    """
    meta = get_metadata_for_nifti(in_file)

    # Process PE direction
    pe_meta = meta.get('PhaseEncodingDirection')
    if pe_meta is None:
        raise RuntimeError('PhaseEncodingDirection metadata not found for '
                           ' %s' % in_file)

    if pe_meta[0] == 'i':
        pe_dir = 0
    elif pe_meta[0] == 'j':
        pe_dir = 1
    elif pe_meta[0] == 'k':
        LOGGER.warn('Detected phase-encoding direction perpendicular '
                    'to the axial plane of the brain.')
        pe_dir = 2

    if pe_meta.endswith('-'):
        pe_dir *= -1.0

    # Option 1: we find the RO time label
    ro_time = meta.get('TotalReadoutTime', None)

    # Option 2: we find the effective echo spacing label
    eff_ec = meta.get('EffectiveEchoSpacing', None)
    if ro_time is None and eff_ec is not None:
        pe_echoes = nb.load(in_file).shape[int(abs(pe_dir))] - 1
        ro_time = eff_ec * pe_echoes

    # Option 3: we find echo time label
    ect = meta.get('EchoTime', None)
    if ro_time is None and ect is not None:
        LOGGER.warn('Total readout time was estimated from the '
                    'EchoTime metadata, please be aware that acceleration '
                    'factors do modify the effective echo time that is '
                    'necessary for fieldmap correction.')
        pe_echoes = nb.load(in_file).shape[int(abs(pe_dir))] - 1
        ro_time = ect * pe_echoes

    # Option 4: using the bandwidth parameter
    pebw = meta.get('BandwidthPerPixelPhaseEncode', None)
    if ro_time is None and pebw is not None:
        LOGGER.warn('Total readout time was estimated from the '
                    'BandwidthPerPixelPhaseEncode metadata, please be aware '
                    'that acceleration factors do modify the effective echo '
                    'time that is necessary for fieldmap correction.')
        pe_echoes = nb.load(in_file).shape[int(abs(pe_dir))]
        ro_time = 1.0 / (pebw * pe_echoes)

    if ro_time is None:
        raise RuntimeError('Readout time could not be set')

    return pe_dir, ro_time


def _run_registration(reference, moving, debug=False, prefix='antsreg', nthreads=None):
    import pkg_resources as pkgr
    from niworkflows.interfaces.registration import ANTSRegistrationRPT as Registration

    ants_settings = pkgr.resource_filename('fmriprep', 'data/fmap-any_registration.json')
    if debug:
        ants_settings = pkgr.resource_filename(
            'fmriprep', 'data/fmap-any_registration_testing.json')
    ants = Registration(from_file=ants_settings,
                        fixed_image=reference,
                        moving_image=moving,
                        output_warped_image=True,
                        output_transform_prefix=prefix)

    if nthreads is not None and nthreads > 0:
        ants.inputs.num_threads = nthreads

    return ants.run().outputs.warped_image

def _coregistration():
    raise NotImplementedError

