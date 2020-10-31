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

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    isdefined,
    traits,
    SimpleInterface,
)


class _GetReadoutTimeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, desc="EPI image corresponding to the metadata")
    metadata = traits.Dict(mandatory=True, desc="metadata corresponding to the inputs")


class _GetReadoutTimeOutputSpec(TraitedSpec):
    readout_time = traits.Float


class GetReadoutTime(SimpleInterface):
    """Calculate the readout time from available metadata."""

    input_spec = _GetReadoutTimeInputSpec
    output_spec = _GetReadoutTimeOutputSpec

    def _run_interface(self, runtime):
        self._results["readout_time"] = get_trt(
            self.inputs.metadata,
            self.inputs.in_file if isdefined(self.inputs.in_file) else None,
        )
        return runtime


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
    import nibabel as nb

    # Use case 1: TRT is defined
    trt = in_meta.get("TotalReadoutTime", None)
    if trt is not None:
        return trt

    # All other cases require the parallel acc and npe (N vox in PE dir)
    acc = float(in_meta.get("ParallelReductionFactorInPlane", 1.0))
    npe = nb.load(in_file).shape[_get_pe_index(in_meta)]
    etl = npe // acc

    # Use case 2: TRT is defined
    ees = in_meta.get("EffectiveEchoSpacing", None)
    if ees is not None:
        return ees * (etl - 1)

    # Use case 3 (philips scans)
    wfs = in_meta.get("WaterFatShift", None)
    if wfs is not None:
        fstrength = in_meta["MagneticFieldStrength"]
        wfd_ppm = 3.4  # water-fat diff in ppm
        g_ratio_mhz_t = 42.57  # gyromagnetic ratio for proton (1H) in MHz/T
        wfs_hz = fstrength * wfd_ppm * g_ratio_mhz_t
        return wfs / wfs_hz

    raise ValueError("Unknown total-readout time specification")


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
    from sdcflows.interfaces.epi import _get_pe_index

    # Use case 1: EES is defined
    ees = in_meta.get("EffectiveEchoSpacing", None)
    if ees is not None:
        return ees

    # All other cases require the parallel acc and npe (N vox in PE dir)
    acc = float(in_meta.get("ParallelReductionFactorInPlane", 1.0))
    npe = nb.load(in_file).shape[_get_pe_index(in_meta)]
    etl = npe // acc

    # Use case 2: TRT is defined
    trt = in_meta.get("TotalReadoutTime", None)
    if trt is not None:
        return trt / (etl - 1)

    # Use case 3 (philips scans)
    wfs = in_meta.get("WaterFatShift", None)
    if wfs is not None:
        fstrength = in_meta["MagneticFieldStrength"]
        wfd_ppm = 3.4  # water-fat diff in ppm
        g_ratio_mhz_t = 42.57  # gyromagnetic ratio for proton (1H) in MHz/T
        wfs_hz = fstrength * wfd_ppm * g_ratio_mhz_t
        return wfs / (wfs_hz * etl)

    raise ValueError("Unknown effective echo-spacing specification")


def _get_pe_index(meta):
    pe = meta["PhaseEncodingDirection"]
    try:
        return {"i": 0, "j": 1, "k": 2}[pe[0]]
    except KeyError:
        raise RuntimeError('"%s" is an invalid PE string' % pe)
