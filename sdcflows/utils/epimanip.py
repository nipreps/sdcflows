"""
Manipulation of EPI data.

.. testsetup::

    >>> tmpdir = getfixture('tmpdir')
    >>> tmp = tmpdir.chdir() # changing to a temporary directory
    >>> nb.Nifti1Image(np.zeros((90, 90, 60)), None, None).to_filename(
    ...     tmpdir.join('epi.nii.gz').strpath)

"""


def get_trt(in_meta, in_file=None):
    r"""
    Obtain the *total readout time* :math:`T_\text{ro}` from available metadata.

    BIDS provides two standard mechanisms to store the total readout time,
    :math:`T_\text{ro}`, of :abbr:`EPI (echo-planar imaging)` scans.
    The first option is that a ``TotalReadoutTime`` field is found
    in the JSON sidecar:

    >>> meta = {'TotalReadoutTime': 0.05251}
    >>> get_trt(meta)
    0.05251

    Alternatively, the *effective echo spacing* :math:`t_\text{ees}`
    (``EffectiveEchoSpacing`` BIDS field) may be provided.
    Then, the total readout time :math:`T_\text{ro}` can be calculated
    as follows:

    .. math ::

        T_\text{ro} = t_\text{ees} \cdot (N_\text{PE} - 1),
        \label{eq:rotime-ees}\tag{1}

    where :math:`N_\text{PE}` is the number of pixels along the
    :abbr:`PE (phase-encoding)` direction **on the reconstructed matrix**.

    >>> meta = {'EffectiveEchoSpacing': 0.00059,
    ...         'PhaseEncodingDirection': 'j-'}
    >>> f"{get_trt(meta, in_file='epi.nii.gz'):g}"
    '0.05251'

    Using nonstandard metadata, there are further options.
    If the *echo spacing* :math:`t_\text{es}` (do not confuse with the
    *effective echo spacing*, :math:`t_\text{ees}`) is set and the
    parallel acceleration factor
    (:abbr:`GRAPPA (GeneRalized Auto-calibrating Partial Parallel Acquisition)`,
    :abbr:`ARC (Auto-calibrating Reconstruction for Cartesian imaging)`, etc.)
    of the EPI :math:`f_\text{acc}` is known, then it is possible to calculate
    the readout time as:

    .. math ::

        T_\text{ro} = t_\text{es} \cdot
            (\left\lfloor\frac{N_\text{PE}}{f_\text{acc}} \right\rfloor - 1).

    >>> meta = {'EchoSpacing': 0.00119341,
    ...         'PhaseEncodingDirection': 'j-',
    ...         'ParallelReductionFactorInPlane': 2}
    >>> f"{get_trt(meta, in_file='epi.nii.gz'):g}"
    '0.05251'

    .. caution::

       Philips stores different parameter names, and there has been quite a bit
       of reverse-engineering and discussion around how to get the total
       readout-time right for the vendor.

       The implementation done here follows the findings of Dr. Rorden,
       summarized in `this post
       <https://github.com/rordenlab/dcm2niix/issues/377#issuecomment-598685590>`__.

    It seems to be possible to calculate the **effective** echo spacing
    (in seconds) as:

    .. math ::

        t_\text{ees} = \frac{f_\text{wfs}}
            {B_0 \gamma \Delta_\text{w/f} \cdot (f_\text{EPI} + 1)},
            \label{eq:philips-ees}\tag{2}

    where :math:`f_\text{wfs}` is the water-fat-shift in pixels,
    :math:`B_0` is the field strength in T, :math:`\gamma` is the
    gyromagnetic ratio, :math:`\Delta_\text{w/f}` is the water/fat
    difference in ppm and :math:`f_\text{EPI}` is Philip's «*EPI factor*,»
    which accounts for in-plane acceleration with :abbr:`SENSE
    (SENSitivity Encoding)`.
    The problem with Philip's «*EPI factor*» is that it is absolutely necessary
    to calculate the effective echo spacing, because the reported SENSE
    acceleration factor does not allow to calculate the effective train
    length from the reconstructed matrix size along the PE direction
    (neither from the acquisition matrix size if it is strangely found
    stored within the metadata).
    For :math:`B_0 = 3.0` [T], then
    :math:`B_0 \gamma \Delta_\text{w/f} \approx 434.215`, as
    in `early discussions held on the FSL listserv
    <https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;162ab1a3.1308>`__.

    As per Dr. Rorden, Eq. :math:`\eqref{eq:philips-ees}` is equivalent to
    the following formulation:

    .. math ::

        t_\text{ees} = \frac{f_\text{wfs}}
            {3.4 \cdot F_\text{img} \cdot (f_\text{EPI} + 1)},

    where :math:`F_\text{img}` is the «*Imaging Frequency*» in MHz,
    as reported by the Philips console.
    This second formulation seems to be preferred for the better accuracy
    of the Imaging Frequency field over the Magnetic field strength.

    Once the effective echo spacing is obtained, the total readout time
    can then be calculated with Eq. :math:`\eqref{eq:rotime-ees}`.

    >>> meta = {'WaterFatShift': 9.2227266,
    ...         'EPIFactor': 35,
    ...         'ImagingFrequency': 127.7325,
    ...         'PhaseEncodingDirection': 'j-'}
    >>> f"{get_trt(meta, in_file='epi.nii.gz'):0.5f}"
    '0.05251'

    >>> meta = {'WaterFatShift': 9.2227266,
    ...         'EPIFactor': 35,
    ...         'MagneticFieldStrength': 3,
    ...         'PhaseEncodingDirection': 'j-'}
    >>> f"{get_trt(meta, in_file='epi.nii.gz'):0.5f}"
    '0.05251'

    If enough metadata is not available, raise an error:

    >>> get_trt({'PhaseEncodingDirection': 'j-'},
    ...         in_file='epi.nii.gz')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError:

    .. admonition:: Thanks

        With thanks to Dr. Rorden for his thourough
        `assessment <https://github.com/rordenlab/dcm2niix/issues/377>`__
        and `validation <https://osf.io/9ucek/>`__ on the matter,
        and to Pravesh Parekh for `his wonderful review on NeuroStars
        <https://neurostars.org/t/consolidating-epi-echo-spacing-and-readout-time-for-philips-scanner/4406>`__.

    .. admonition:: See Also

        Some useful links regarding the calculation of the readout time for Philips:

          * `Brain Voyager documentation
            <https://support.brainvoyager.com/brainvoyager/functional-analysis-preparation/29-pre-processing/78-epi-distortion-correction-echo-spacing-and-bandwidth>`__
            -- Please note that I (OE) *believe* the statement about the effective echo-spacing
            on Philips **is wrong**, as the EPI factor should account for the in-plane
            acceleration.
          * `Disappeared documentation of the Spinoza Center
            <https://web.archive.org/web/20130420035502/www.spinozacentre.nl/wiki/index.php/NeuroWiki:Current_developments>`__.
          * This `guide for preprocessing of EPI data <https://osf.io/hks7x/>`__.

    """
    import nibabel as nb

    # Use case 1: TRT is defined
    if "TotalReadoutTime" in in_meta:
        return in_meta.get("TotalReadoutTime")

    # npe = N voxels PE direction
    pe_index = "ijk".index(in_meta["PhaseEncodingDirection"][0])
    npe = nb.load(in_file).shape[pe_index]

    # Use case 2: EES is defined
    ees = in_meta.get("EffectiveEchoSpacing")
    if ees:
        # Effective echo spacing means that acceleration factors have been accounted for.
        return ees * (npe - 1)

    try:
        echospacing = in_meta["EchoSpacing"]
        acc_factor = in_meta["ParallelReductionFactorInPlane"]
    except KeyError:
        pass
    else:
        # etl = effective train length
        etl = npe // acc_factor
        return echospacing * (etl - 1)

    # Use case 3 (Philips scans)
    try:
        wfs = in_meta["WaterFatShift"]
        epifactor = in_meta["EPIFactor"]
    except KeyError:
        pass
    else:
        wfs_hz = (
            (in_meta.get("ImagingFrequency", 0) * 3.39941)
            or (in_meta.get("MagneticFieldStrength", 0) * 144.7383333)
            or None
        )
        if wfs_hz:
            ees = wfs / (wfs_hz * (epifactor + 1))
            return ees * (npe - 1)

    raise ValueError("Unknown total-readout time specification")
