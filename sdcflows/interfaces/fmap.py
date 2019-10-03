# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces to deal with the various types of fieldmap sources.

    .. testsetup::

        >>> tmpdir = getfixture('tmpdir')
        >>> tmp = tmpdir.chdir() # changing to a temporary directory
        >>> nb.Nifti1Image(np.zeros((90, 90, 60)), None, None).to_filename(
        ...     tmpdir.join('epi.nii.gz').strpath)


"""

import numpy as np
import nibabel as nb
from copy import deepcopy
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, TraitedSpec, File, isdefined, traits,
    SimpleInterface)

LOGGER = logging.getLogger('nipype.interface')

<<<<<<< HEAD
<<<<<<< HEAD
class _ProcessPhasesInputSpec(BaseInterfaceInputSpec):
=======
class ProcessPhasesInputSpec(BaseInterfaceInputSpec):
>>>>>>> start workflow for calculating phasediff from phase images
=======
class _ProcessPhasesInputSpec(BaseInterfaceInputSpec):
>>>>>>> Update sdcflows/interfaces/fmap.py
    phase1_file = File(exists=True, mandatory=True, desc='phase1 file')
    phase2_file = File(exists=True, mandatory=True, desc='phase2 file')
    phase1_metadata = traits.Dict(mandatory=True, desc='phase1 metadata dict')
    phase2_metadata = traits.Dict(mandatory=True, desc='phase2 metadata dict')
<<<<<<< HEAD
=======


class _ProcessPhasesOutputSpec(TraitedSpec):
    short_te_phase_image = File(exists=True, desc='short TE phase image scaled for unwrapping')
    short_te_phase_metadata = traits.Dict(desc='short TE phase image metadata')
    long_te_phase_image = File(exists=True, desc='long TE phase image scaled for unwrapping')
    long_te_phase_metadata = traits.Dict(desc='long TE phase image metadata')
    phasediff_metadata = traits.Dict(desc='the phasediff metadata')


class ProcessPhases(SimpleInterface):
    """
    Process phase1, phase2 images so they can be unwrapped.

    Steps are taken from https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide
    """
    input_spec = _ProcessPhasesInputSpec
    output_spec = _ProcessPhasesOutputSpec

    def _run_interface(self, runtime):
        images = [self.inputs.phase1_file, self.inputs.phase2_file]
        metadatas = [self.inputs.phase1_metadata, self.inputs.phase2_metadata]
        echo_times = [meta.get("EchoTime") for meta in metadatas]
        if None in echo_times or echo_times[0] == echo_times[1]:
            raise ValueError('Echo time metadata are missing or invalid')

        # Order the images by echo time
        short_echo_index = echo_times.index(min(echo_times))
        long_echo_index = echo_times.index(max(echo_times))
        short_echo_image = images[short_echo_index]
        long_echo_image = images[long_echo_index]

        # Rescale and save the short TE image
        se_phase_image = nb.load(short_echo_image)
        se_phase_data = se_phase_image.get_fdata()
        se_rescaled_data = rescale_phase_image(se_phase_data)
        se_rescaled_output = fname_presuffix(short_echo_image, suffix='_shortTE_scaled',
                                             newpath=runtime.cwd)
        nb.Nifti1Image(se_rescaled_data, se_phase_image.affine, se_phase_image.header
                       ).to_filename(se_rescaled_output)
        self._results['short_te_phase_image'] = se_rescaled_output
        self._results['short_te_phase_metadata'] = metadatas[short_echo_index]

        # Rescale and save the long TE image
        le_phase_image = nb.load(long_echo_image)
        le_phase_data = le_phase_image.get_fdata()
        le_rescaled_data = rescale_phase_image(le_phase_data)
        le_rescaled_output = fname_presuffix(long_echo_image, suffix='_longTE_scaled',
                                             newpath=runtime.cwd)
        nb.Nifti1Image(le_rescaled_data, le_phase_image.affine, le_phase_image.header
                       ).to_filename(le_rescaled_output)
        self._results['long_te_phase_image'] = le_rescaled_output
        self._results['long_te_phase_metadata'] = metadatas[long_echo_index]

        merged_metadata = deepcopy(metadatas[0])
        del merged_metadata['EchoTime']
        merged_metadata['EchoTime1'] = float(echo_times[short_echo_index])
        merged_metadata['EchoTime2'] = float(echo_times[long_echo_index])
        self._results['phasediff_metadata'] = merged_metadata
        return runtime


def rescale_phase_image(phase_data):
    """Ensure that phase images are in a usable range for unwrapping.

    From the FUGUE User guide::

        If you have seperate phase volumes that are in integer format then do:

        fslmaths orig_phase0 -mul 3.14159 -div 2048 phase0_rad -odt float
        fslmaths orig_phase1 -mul 3.14159 -div 2048 phase1_rad -odt float

        Note that the value of 2048 needs to be adjusted for each different
        site/scanner/sequence in order to be correct. The final range of the
        phase0_rad image should be approximately 0 to 6.28. If this is not the
        case then this scaling is wrong. If you have separate phase volumes are
        not in integer format, you must still check that the units are in radians,
        and if not scale them appropriately using fslmaths.
    """
    imax = phase_data.max()
    imin = phase_data.min()
    scaled = (phase_data - imin) / (imax - imin)
    return 2 * np.pi * scaled


class PhaseDiffInputSpec(BaseInterfaceInputSpec):
    phasediff = File(exists=True, desc='single-volume phasediff image')
    phase1 = File(exists=True, desc='single-volume phase image')
    phase2 = File(exists=True, desc='single-volume phase image')
    metadata = traits.Dict(mandatory=True)


class PhaseDiffOutputSpec(TraitedSpec):
    phasediff = File(exists=True, mandatory=True)


class PhaseDiff(SimpleInterface):
    input_spec = PhaseDiffInputSpec
    output_spec = PhaseDiffOutputSpec

>>>>>>> start workflow for calculating phasediff from phase images


class _ProcessPhasesOutputSpec(TraitedSpec):
    short_te_phase_image = File(exists=True, desc='short TE phase image scaled for unwrapping')
    short_te_phase_metadata = traits.Dict(desc='short TE phase image metadata')
    long_te_phase_image = File(exists=True, desc='long TE phase image scaled for unwrapping')
    long_te_phase_metadata = traits.Dict(desc='long TE phase image metadata')
    phasediff_metadata = traits.Dict(desc='the phasediff metadata')


class ProcessPhases(SimpleInterface):
    """
    Process phase1, phase2 images so they can be unwrapped.

    Steps are taken from https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide
    """
    input_spec = _ProcessPhasesInputSpec
    output_spec = _ProcessPhasesOutputSpec

    def _run_interface(self, runtime):
        images = [self.inputs.phase1_file, self.inputs.phase2_file]
        metadatas = [self.inputs.phase1_metadata, self.inputs.phase2_metadata]
        echo_times = [meta.get("EchoTime") for meta in metadatas]
        if None in echo_times or echo_times[0] == echo_times[1]:
            raise ValueError('Echo time metadata are missing or invalid')

        # Order the images by echo time
        short_echo_index = echo_times.index(min(echo_times))
        long_echo_index = echo_times.index(max(echo_times))
        short_echo_image = images[short_echo_index]
        long_echo_image = images[long_echo_index]

        # Rescale and save the short TE image
        se_phase_image = nb.load(short_echo_image)
        se_phase_data = se_phase_image.get_fdata()
        se_rescaled_data = rescale_phase_image(se_phase_data)
        se_rescaled_output = fname_presuffix(short_echo_image, suffix='_shortTE_scaled',
                                             newpath=runtime.cwd)
        nb.Nifti1Image(se_rescaled_data, se_phase_image.affine, se_phase_image.header
                       ).to_filename(se_rescaled_output)
        self._results['short_te_phase_image'] = se_rescaled_output
        self._results['short_te_phase_metadata'] = metadatas[short_echo_index]

        # Rescale and save the long TE image
        le_phase_image = nb.load(long_echo_image)
        le_phase_data = le_phase_image.get_fdata()
        le_rescaled_data = rescale_phase_image(le_phase_data)
        le_rescaled_output = fname_presuffix(long_echo_image, suffix='_longTE_scaled',
                                             newpath=runtime.cwd)
        nb.Nifti1Image(le_rescaled_data, le_phase_image.affine, le_phase_image.header
                       ).to_filename(le_rescaled_output)
        self._results['long_te_phase_image'] = le_rescaled_output
        self._results['long_te_phase_metadata'] = metadatas[long_echo_index]

        merged_metadata = deepcopy(metadatas[0])
        del merged_metadata['EchoTime']
        merged_metadata['EchoTime1'] = float(echo_times[short_echo_index])
        merged_metadata['EchoTime2'] = float(echo_times[long_echo_index])
        self._results['phasediff_metadata'] = merged_metadata
        return runtime


def rescale_phase_image(phase_data):
    """Ensure that phase images are in a usable range for unwrapping.

    From the FUGUE User guide::

        If you have seperate phase volumes that are in integer format then do:

        fslmaths orig_phase0 -mul 3.14159 -div 2048 phase0_rad -odt float
        fslmaths orig_phase1 -mul 3.14159 -div 2048 phase1_rad -odt float

        Note that the value of 2048 needs to be adjusted for each different
        site/scanner/sequence in order to be correct. The final range of the
        phase0_rad image should be approximately 0 to 6.28. If this is not the
        case then this scaling is wrong. If you have separate phase volumes are
        not in integer format, you must still check that the units are in radians,
        and if not scale them appropriately using fslmaths.
    """
    imax = phase_data.max()
    imin = phase_data.min()
    scaled = (phase_data - imin) / (imax - imin)
    return 2 * np.pi * scaled


class PhaseDiffInputSpec(BaseInterfaceInputSpec):
    phasediff = File(exists=True, desc='single-volume phasediff image')
    phase1 = File(exists=True, desc='single-volume phase image')
    phase2 = File(exists=True, desc='single-volume phase image')
    metadata = traits.Dict(mandatory=True)


class PhaseDiffOutputSpec(TraitedSpec):
    phasediff = File(exists=True, mandatory=True)


class PhaseDiff(SimpleInterface):
    input_spec = PhaseDiffInputSpec
    output_spec = PhaseDiffOutputSpec


class _FieldEnhanceInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    in_mask = File(exists=True, desc='brain mask')
    in_magnitude = File(exists=True, desc='input magnitude')
    unwrap = traits.Bool(False, usedefault=True, desc='run phase unwrap')
    despike = traits.Bool(True, usedefault=True, desc='run despike filter')
    bspline_smooth = traits.Bool(True, usedefault=True, desc='run 3D bspline smoother')
    mask_erode = traits.Int(1, usedefault=True, desc='mask erosion iterations')
    despike_threshold = traits.Float(0.2, usedefault=True, desc='mask erosion iterations')
    num_threads = traits.Int(1, usedefault=True, nohash=True, desc='number of jobs')


class _FieldEnhanceOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')
    out_unwrapped = File(desc='unwrapped fieldmap')


class FieldEnhance(SimpleInterface):
    """Massage the input fieldmap (masking, despiking, etc.)."""

    input_spec = _FieldEnhanceInputSpec
    output_spec = _FieldEnhanceOutputSpec

    def _run_interface(self, runtime):
        from scipy import ndimage as sim

        fmap_nii = nb.load(self.inputs.in_file)
        data = np.squeeze(fmap_nii.get_data().astype(np.float32))

        # Despike / denoise (no-mask)
        if self.inputs.despike:
            data = _despike2d(data, self.inputs.despike_threshold)

        mask = None
        if isdefined(self.inputs.in_mask):
            masknii = nb.load(self.inputs.in_mask)
            mask = masknii.get_data().astype(np.uint8)

            # Dilate mask
            if self.inputs.mask_erode > 0:
                struc = sim.iterate_structure(sim.generate_binary_structure(3, 2), 1)
                mask = sim.binary_erosion(
                    mask, struc,
                    iterations=self.inputs.mask_erode
                ).astype(np.uint8)  # pylint: disable=no-member

        self._results['out_file'] = fname_presuffix(
            self.inputs.in_file, suffix='_enh', newpath=runtime.cwd)
        datanii = nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header)

        if self.inputs.unwrap:
            data = _unwrap(data, self.inputs.in_magnitude, mask)
            self._results['out_unwrapped'] = fname_presuffix(
                self.inputs.in_file, suffix='_unwrap', newpath=runtime.cwd)
            nb.Nifti1Image(data, fmap_nii.affine, fmap_nii.header).to_filename(
                self._results['out_unwrapped'])

        if not self.inputs.bspline_smooth:
            datanii.to_filename(self._results['out_file'])
            return runtime
        else:
            from ..utils import bspline as fbsp
            from statsmodels.robust.scale import mad

            # Fit BSplines (coarse)
            bspobj = fbsp.BSplineFieldmap(datanii, weights=mask,
                                          njobs=self.inputs.num_threads)
            bspobj.fit()
            smoothed1 = bspobj.get_smoothed()

            # Manipulate the difference map
            diffmap = data - smoothed1.get_data()
            sderror = mad(diffmap[mask > 0])
            LOGGER.info('SD of error after B-Spline fitting is %f', sderror)
            errormask = np.zeros_like(diffmap)
            errormask[np.abs(diffmap) > (10 * sderror)] = 1
            errormask *= mask

            nslices = 0
            try:
                errorslice = np.squeeze(np.argwhere(errormask.sum(0).sum(0) > 0))
                nslices = errorslice[-1] - errorslice[0]
            except IndexError:  # mask is empty, do not refine
                pass

            if nslices > 1:
                diffmapmsk = mask[..., errorslice[0]:errorslice[-1]]
                diffmapnii = nb.Nifti1Image(
                    diffmap[..., errorslice[0]:errorslice[-1]] * diffmapmsk,
                    datanii.affine, datanii.header)

                bspobj2 = fbsp.BSplineFieldmap(diffmapnii, knots_zooms=[24., 24., 4.],
                                               njobs=self.inputs.num_threads)
                bspobj2.fit()
                smoothed2 = bspobj2.get_smoothed().get_data()

                final = smoothed1.get_data().copy()
                final[..., errorslice[0]:errorslice[-1]] += smoothed2
            else:
                final = smoothed1.get_data()

            nb.Nifti1Image(final, datanii.affine, datanii.header).to_filename(
                self._results['out_file'])

        return runtime


class _FieldToRadSInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    fmap_range = traits.Float(desc='range of input field map')


class _FieldToRadSOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')
    fmap_range = traits.Float(desc='range of input field map')


class FieldToRadS(SimpleInterface):
    """Convert from arbitrary units to rad/s."""

    input_spec = _FieldToRadSInputSpec
    output_spec = _FieldToRadSOutputSpec

    def _run_interface(self, runtime):
        fmap_range = None
        if isdefined(self.inputs.fmap_range):
            fmap_range = self.inputs.fmap_range
        self._results['out_file'], self._results['fmap_range'] = _torads(
            self.inputs.in_file, fmap_range, newpath=runtime.cwd)
        return runtime


class _FieldToHzInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    range_hz = traits.Float(mandatory=True, desc='range of input field map')


class _FieldToHzOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')


class FieldToHz(SimpleInterface):
    """Convert from arbitrary units to Hz."""

    input_spec = _FieldToHzInputSpec
    output_spec = _FieldToHzOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = _tohz(
            self.inputs.in_file, self.inputs.range_hz, newpath=runtime.cwd)
        return runtime


class _Phasediff2FieldmapInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='input fieldmap')
    metadata = traits.Dict(mandatory=True, desc='BIDS metadata dictionary')


class _Phasediff2FieldmapOutputSpec(TraitedSpec):
    out_file = File(desc='the output fieldmap')


class Phasediff2Fieldmap(SimpleInterface):
    """Convert a phase difference map into a fieldmap in Hz."""

    input_spec = _Phasediff2FieldmapInputSpec
    output_spec = _Phasediff2FieldmapOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = phdiff2fmap(
            self.inputs.in_file,
            _delta_te(self.inputs.metadata),
            newpath=runtime.cwd)
        return runtime


def _despike2d(data, thres, neigh=None):
    """Despike axial slices, as done in FSL's ``epiunwarp``."""
    if neigh is None:
        neigh = [-1, 0, 1]
    nslices = data.shape[-1]

    for k in range(nslices):
        data2d = data[..., k]

        for i in range(data2d.shape[0]):
            for j in range(data2d.shape[1]):
                vals = []
                thisval = data2d[i, j]
                for ii in neigh:
                    for jj in neigh:
                        try:
                            vals.append(data2d[i + ii, j + jj])
                        except IndexError:
                            pass
                vals = np.array(vals)
                patch_range = vals.max() - vals.min()
                patch_med = np.median(vals)

                if (patch_range > 1e-6 and
                        (abs(thisval - patch_med) / patch_range) > thres):
                    data[i, j, k] = patch_med
    return data


def _unwrap(fmap_data, mag_file, mask=None):
    from math import pi
    from nipype.interfaces.fsl import PRELUDE
    magnii = nb.load(mag_file)

    if mask is None:
        mask = np.ones_like(fmap_data, dtype=np.uint8)

    fmapmax = max(abs(fmap_data[mask > 0].min()), fmap_data[mask > 0].max())
    fmap_data *= pi / fmapmax

    nb.Nifti1Image(fmap_data, magnii.affine).to_filename('fmap_rad.nii.gz')
    nb.Nifti1Image(mask, magnii.affine).to_filename('fmap_mask.nii.gz')
    nb.Nifti1Image(magnii.get_data(), magnii.affine).to_filename('fmap_mag.nii.gz')

    # Run prelude
    res = PRELUDE(phase_file='fmap_rad.nii.gz',
                  magnitude_file='fmap_mag.nii.gz',
                  mask_file='fmap_mask.nii.gz').run()

    unwrapped = nb.load(res.outputs.unwrapped_phase_file).get_data() * (fmapmax / pi)
    return unwrapped


def get_ees(in_meta, in_file=None):
    r"""
    Extract the *effective echo spacing* :math:`t_\text{ees}` from BIDS.

    Calculate the *effective echo spacing* :math:`t_\text{ees}`
    for an input :abbr:`EPI (echo-planar imaging)` scan.


    There are several procedures to calculate the effective
    echo spacing. The basic one is that an ``EffectiveEchoSpacing``
    field is set in the JSON sidecar. The following examples
    use an ``'epi.nii.gz'`` file-stub which has 90 pixels in the
    j-axis encoding direction.

    >>> meta = {'EffectiveEchoSpacing': 0.00059,
    ...         'PhaseEncodingDirection': 'j-'}
    >>> get_ees(meta)
    0.00059

    If the *total readout time* :math:`T_\text{ro}` (``TotalReadoutTime``
    BIDS field) is provided, then the effective echo spacing can be
    calculated reading the number of voxels :math:`N_\text{PE}` along the
    readout direction and the parallel acceleration
    factor of the EPI

      .. math ::

           =  T_\text{ro} \,  (N_\text{PE} / f_\text{acc} - 1)^{-1}

    where :math:`N_y` is the number of pixels along the phase-encoding direction
    :math:`y`, and :math:`f_\text{acc}` is the parallel imaging acceleration factor
    (:abbr:`GRAPPA (GeneRalized Autocalibrating Partial Parallel Acquisition)`,
    :abbr:`ARC (Autocalibrating Reconstruction for Cartesian imaging)`, etc.).

    >>> meta = {'TotalReadoutTime': 0.02596,
    ...         'PhaseEncodingDirection': 'j-',
    ...         'ParallelReductionFactorInPlane': 2}
    >>> get_ees(meta, in_file='epi.nii.gz')
    0.00059

    Some vendors, like Philips, store different parameter names (see
    http://dbic.dartmouth.edu/pipermail/mrusers/attachments/20141112/eb1d20e6/attachment.pdf
    ):

    >>> meta = {'WaterFatShift': 8.129,
    ...         'MagneticFieldStrength': 3,
    ...         'PhaseEncodingDirection': 'j-',
    ...         'ParallelReductionFactorInPlane': 2}
    >>> get_ees(meta, in_file='epi.nii.gz')
    0.00041602630141921826

    """

    import nibabel as nb
    from sdcflows.interfaces.fmap import _get_pe_index

    # Use case 1: EES is defined
    ees = in_meta.get('EffectiveEchoSpacing', None)
    if ees is not None:
        return ees

    # All other cases require the parallel acc and npe (N vox in PE dir)
    acc = float(in_meta.get('ParallelReductionFactorInPlane', 1.0))
    npe = nb.load(in_file).shape[_get_pe_index(in_meta)]
    etl = npe // acc

    # Use case 2: TRT is defined
    trt = in_meta.get('TotalReadoutTime', None)
    if trt is not None:
        return trt / (etl - 1)

    # Use case 3 (philips scans)
    wfs = in_meta.get('WaterFatShift', None)
    if wfs is not None:
        fstrength = in_meta['MagneticFieldStrength']
        wfd_ppm = 3.4  # water-fat diff in ppm
        g_ratio_mhz_t = 42.57  # gyromagnetic ratio for proton (1H) in MHz/T
        wfs_hz = fstrength * wfd_ppm * g_ratio_mhz_t
        return wfs / (wfs_hz * etl)

    raise ValueError('Unknown effective echo-spacing specification')


def get_trt(in_meta, in_file=None):
    r"""
    Extract the *total readout time* :math:`t_\text{RO}` from BIDS.

    Calculate the *total readout time* for an input
    :abbr:`EPI (echo-planar imaging)` scan.

    There are several procedures to calculate the total
    readout time. The basic one is that a ``TotalReadoutTime``
    field is set in the JSON sidecar. The following examples
    use an ``'epi.nii.gz'`` file-stub which has 90 pixels in the
    j-axis encoding direction.

    >>> meta = {'TotalReadoutTime': 0.02596}
    >>> get_trt(meta)
    0.02596

    If the *effective echo spacing* :math:`t_\text{ees}`
    (``EffectiveEchoSpacing`` BIDS field) is provided, then the
    total readout time can be calculated reading the number
    of voxels along the readout direction :math:`T_\text{ro}`
    and the parallel acceleration factor of the EPI :math:`f_\text{acc}`.

      .. math ::

          T_\text{ro} = t_\text{ees} \, (N_\text{PE} / f_\text{acc} - 1)

    >>> meta = {'EffectiveEchoSpacing': 0.00059,
    ...         'PhaseEncodingDirection': 'j-',
    ...         'ParallelReductionFactorInPlane': 2}
    >>> get_trt(meta, in_file='epi.nii.gz')
    0.02596

    Some vendors, like Philips, store different parameter names:

    >>> meta = {'WaterFatShift': 8.129,
    ...         'MagneticFieldStrength': 3,
    ...         'PhaseEncodingDirection': 'j-',
    ...         'ParallelReductionFactorInPlane': 2}
    >>> get_trt(meta, in_file='epi.nii.gz')
    0.018721183563864822

    """
    # Use case 1: TRT is defined
    trt = in_meta.get('TotalReadoutTime', None)
    if trt is not None:
        return trt

    # All other cases require the parallel acc and npe (N vox in PE dir)
    acc = float(in_meta.get('ParallelReductionFactorInPlane', 1.0))
    npe = nb.load(in_file).shape[_get_pe_index(in_meta)]
    etl = npe // acc

    # Use case 2: TRT is defined
    ees = in_meta.get('EffectiveEchoSpacing', None)
    if ees is not None:
        return ees * (etl - 1)

    # Use case 3 (philips scans)
    wfs = in_meta.get('WaterFatShift', None)
    if wfs is not None:
        fstrength = in_meta['MagneticFieldStrength']
        wfd_ppm = 3.4  # water-fat diff in ppm
        g_ratio_mhz_t = 42.57  # gyromagnetic ratio for proton (1H) in MHz/T
        wfs_hz = fstrength * wfd_ppm * g_ratio_mhz_t
        return wfs / wfs_hz

    raise ValueError('Unknown total-readout time specification')


def _get_pe_index(meta):
    pe = meta['PhaseEncodingDirection']
    try:
        return {'i': 0, 'j': 1, 'k': 2}[pe[0]]
    except KeyError:
        raise RuntimeError('"%s" is an invalid PE string' % pe)


def _torads(in_file, fmap_range=None, newpath=None):
    """
    Convert a field map to rad/s units.

    If fmap_range is None, the range of the fieldmap
    will be automatically calculated.

    Use fmap_range=0.5 to convert from Hz to rad/s

    """
    from math import pi
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(in_file, suffix='_rad', newpath=newpath)
    fmapnii = nb.load(in_file)
    fmapdata = fmapnii.get_data()

    if fmap_range is None:
        fmap_range = max(abs(fmapdata.min()), fmapdata.max())
    fmapdata = fmapdata * (pi / fmap_range)
    out_img = nb.Nifti1Image(fmapdata, fmapnii.affine, fmapnii.header)
    out_img.set_data_dtype('float32')
    out_img.to_filename(out_file)
    return out_file, fmap_range


def _tohz(in_file, range_hz, newpath=None):
    """Convert a field map to Hz units."""
    from math import pi
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    out_file = fname_presuffix(in_file, suffix='_hz', newpath=newpath)
    fmapnii = nb.load(in_file)
    fmapdata = fmapnii.get_data()
    fmapdata = fmapdata * (range_hz / pi)
    out_img = nb.Nifti1Image(fmapdata, fmapnii.affine, fmapnii.header)
    out_img.set_data_dtype('float32')
    out_img.to_filename(out_file)
    return out_file


def phdiff2fmap(in_file, delta_te, newpath=None):
    r"""
    Convert the input phase-difference map into a fieldmap in Hz.

    Uses eq. (1) of [Hutton2002]_:

    .. math::

        \Delta B_0 (\text{T}^{-1}) = \frac{\Delta \Theta}{2\pi\gamma \Delta\text{TE}}


    In this case, we do not take into account the gyromagnetic ratio of the
    proton (:math:`\gamma`), since it will be applied inside TOPUP:

    .. math::

        \Delta B_0 (\text{Hz}) = \frac{\Delta \Theta}{2\pi \Delta\text{TE}}

    References
    ----------
    .. [Hutton2002] Hutton et al., Image Distortion Correction in fMRI: A Quantitative
      Evaluation, NeuroImage 16(1):217-240, 2002. doi:`10.1006/nimg.2001.1054
      <https://doi.org/10.1006/nimg.2001.1054>`_.


    """
    import math
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    #  GYROMAG_RATIO_H_PROTON_MHZ = 42.576

    out_file = fname_presuffix(in_file, suffix='_fmap', newpath=newpath)
    image = nb.load(in_file)
    data = (image.get_data().astype(np.float32) / (2. * math.pi * delta_te))
    nii = nb.Nifti1Image(data, image.affine, image.header)
    nii.set_data_dtype(np.float32)
    nii.to_filename(out_file)
    return out_file


def _delta_te(in_values, te1=None, te2=None):
    r"""Read :math:`\Delta_\text{TE}` from BIDS metadata dict."""
    if isinstance(in_values, float):
        te2 = in_values
        te1 = 0.

    if isinstance(in_values, dict):
        te1 = in_values.get('EchoTime1')
        te2 = in_values.get('EchoTime2')

        if not all((te1, te2)):
            te2 = in_values.get('EchoTimeDifference')
            te1 = 0

    if isinstance(in_values, list):
        te2, te1 = in_values
        if isinstance(te1, list):
            te1 = te1[1]
        if isinstance(te2, list):
            te2 = te2[1]

    # For convienience if both are missing we should give one error about them
    if te1 is None and te2 is None:
        raise RuntimeError('EchoTime1 and EchoTime2 metadata fields not found. '
                           'Please consult the BIDS specification.')
    if te1 is None:
        raise RuntimeError(
            'EchoTime1 metadata field not found. Please consult the BIDS specification.')
    if te2 is None:
        raise RuntimeError(
            'EchoTime2 metadata field not found. Please consult the BIDS specification.')

    return abs(float(te2) - float(te1))
