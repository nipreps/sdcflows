# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Processing phase-difference and *directly measured* B0 maps.

.. _sdc_phasediff :

Phase-difference B0 estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The inhomogeneity of the :math:`B_0` field inside the scanner at each voxel is
proportional to the phase drift (at that voxel) between two subsequent
:abbr:`GRE (gradient-recalled echo)` acquisitions.

This corresponds to `this section of the BIDS specification
<https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#two-phase-images-and-two-magnitude-images>`__.
Some scanners produce one ``phasediff`` map, where the drift between the two echos has
already been calulated (see `the corresponding section of BIDS
<https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#case-1-phase-difference-map-and-at-least-one-magnitude-image>`__).

.. _sdc_direct_b0 :

Direct B0 mapping sequences
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The inhomogeneity of the :math:`B_0` field can directly be mapped with a
some MR schemes (such as :abbr:`SE (spiral echo)`).
These *fieldmaps* are described with more detail `here
<https://cni.stanford.edu/wiki/GE_Processing#Fieldmaps>`__.

This corresponds to `this section of the BIDS specification
<https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/01-magnetic-resonance-imaging-data.html#case-3-direct-field-mapping>`__.


"""

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


def init_fmap_wf(omp_nthreads, mode="phasediff", name="fmap_wf"):
    """
    Estimate the fieldmap based on a field-mapping MRI acquisition.

    Estimates the fieldmap using either one phase-difference map or
    image and one or more
    magnitude images corresponding to two or more :abbr:`GRE (Gradient Echo sequence)`
    acquisitions.

    When we have a sequence that directly measures the fieldmap,
    we just need to mask it (using the corresponding magnitude image)
    to remove the noise in the surrounding air region, and ensure that
    units are Hz.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.fieldmap import init_fmap_wf
            wf = init_fmap_wf(omp_nthreads=6)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use.
    name : :obj:`str`
        Unique name of this workflow.

    Inputs
    ------
    magnitude : :obj:`str`
        Path to the corresponding magnitude image for anatomical reference.
    fieldmap : :obj:`str`
        Path to the fieldmap acquisition (``*_fieldmap.nii[.gz]`` of BIDS).

    Outputs
    -------
    fmap : :obj:`str`
        Path to the estimated fieldmap.
    fmap_ref : :obj:`str`
        Path to a preprocessed magnitude image reference.
    fmap_mask : :obj:`str`
        Path to a binary brain mask corresponding to the ``fmap`` and ``fmap_ref``
        pair.

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["magnitude", "fieldmap"]), name="inputnode"
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fmap", "fmap_ref", "fmap_mask"]),
        name="outputnode",
    )

    magnitude_wf = init_magnitude_wf(omp_nthreads=omp_nthreads)
    fmap_postproc_wf = init_fmap_postproc_wf(omp_nthreads=omp_nthreads)

    # fmt: off
    workflow.connect([
        (inputnode, magnitude_wf, [("magnitude", "inputnode.magnitude")]),
        (magnitude_wf, fmap_postproc_wf, [
            ("outputnode.fmap_mask", "inputnode.fmap_mask"),
            ("outputnode.fmap_ref", "inputnode.fmap_ref")]),
        (magnitude_wf, outputnode, [
            ("outputnode.fmap_mask", "fmap_mask"),
            ("outputnode.fmap_ref", "fmap_ref"),
        ]),
        (fmap_postproc_wf, outputnode, [("outputnode.out_fmap", "fmap")]),
    ])
    # fmt: on

    if mode == "phasediff":
        workflow.__desc__ = """\
A *B<sub>0</sub>* nonuniformity map (or *fieldmap*) was estimated from the
phase-drift map(s) measure with two consecutive GRE (gradient-recall echo)
acquisitions.
"""
        phdiff_wf = init_phdiff_wf(omp_nthreads)

        # fmt: off
        workflow.connect([
            (inputnode, phdiff_wf, [("fieldmap", "inputnode.phase")]),
            (magnitude_wf, phdiff_wf, [
                ("outputnode.fmap_ref", "inputnode.magnitude"),
                ("outputnode.fmap_mask", "inputnode.mask"),
            ]),
            (phdiff_wf, fmap_postproc_wf, [
                ("outputnode.fieldmap", "inputnode.fmap"),
            ]),
        ])
        # fmt: on
    else:
        from niworkflows.interfaces.nibabel import ApplyMask
        from niworkflows.interfaces.images import IntraModalMerge
        workflow.__desc__ = """\
A *B<sub>0</sub>* nonuniformity map (or *fieldmap*) was directly measured with
an MRI scheme designed with that purpose (e.g., a spiral pulse sequence).
"""
        # Merge input fieldmap images
        fmapmrg = pe.Node(IntraModalMerge(zero_based_avg=False, hmc=False), name="fmapmrg")
        applymsk = pe.Node(ApplyMask(), name="applymsk")
        # fmt: off
        workflow.connect([
            (inputnode, fmapmrg, [("fieldmap", "in_files")]),
            (fmapmrg, applymsk, [("out_avg", "in_file")]),
            (magnitude_wf, applymsk, [("outputnode.fmap_mask", "in_mask")]),
            (applymsk, fmap_postproc_wf, [("out_file", "inputnode.fmap")]),
        ])
        # fmt: on

    return workflow


def init_magnitude_wf(omp_nthreads, name="magnitude_wf"):
    """
    Prepare the magnitude part of :abbr:`GRE (gradient-recalled echo)` fieldmaps.

    Average (if not done already) the magnitude part of the
    :abbr:`GRE (gradient recalled echo)` images, run N4 to
    correct for B1 field nonuniformity, and skull-strip the
    preprocessed magnitude.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.fieldmap import init_magnitude_wf
            wf = init_magnitude_wf(omp_nthreads=6)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    name : :obj:`str`
        Name of workflow (default: ``prepare_magnitude_w``)

    Inputs
    ------
    magnitude : :obj:`os.PathLike`
        Path to the corresponding magnitude path(s).

    Outputs
    -------
    fmap_ref : :obj:`os.PathLike`
        Path to the fieldmap reference calculated in this workflow.
    fmap_mask : :obj:`os.PathLike`
        Path to a binary brain mask corresponding to the reference above.

    """
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from niworkflows.interfaces.masks import BETRPT
    from niworkflows.interfaces.images import IntraModalMerge

    workflow = Workflow(name=name)
    inputnode = pe.Node(niu.IdentityInterface(fields=["magnitude"]), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fmap_ref", "fmap_mask", "mask_report"]),
        name="outputnode",
    )

    # Merge input magnitude images
    # Do not reorient to RAS to preserve the validity of PhaseEncodingDirection
    magmrg = pe.Node(IntraModalMerge(hmc=False, to_ras=False), name="magmrg")

    # de-gradient the fields ("bias/illumination artifact")
    n4_correct = pe.Node(
        N4BiasFieldCorrection(dimension=3, copy_header=True),
        name="n4_correct",
        n_procs=omp_nthreads,
    )
    bet = pe.Node(BETRPT(generate_report=True, frac=0.6, mask=True), name="bet")

    # fmt: off
    workflow.connect([
        (inputnode, magmrg, [("magnitude", "in_files")]),
        (magmrg, n4_correct, [("out_avg", "input_image")]),
        (n4_correct, bet, [("output_image", "in_file")]),
        (bet, outputnode, [("mask_file", "fmap_mask"),
                           ("out_file", "fmap_ref"),
                           ("out_report", "mask_report")]),
    ])
    # fmt: on
    return workflow


def init_fmap_postproc_wf(
    omp_nthreads, median_kernel_size=5, name="fmap_postproc_wf"
):
    """
    Postprocess a :math:`B_0` map estimated elsewhere.

    This workflow denoises (mostly via smoothing) a :math:`B_0` fieldmap.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.fieldmap import init_fmap_postproc_wf
            wf = init_fmap_postproc_wf(omp_nthreads=6)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    median_kernel_size : :obj:`int`
        Size of the kernel when smoothing is done with a median filter.
    name : :obj:`str`
        Name of workflow (default: ``fmap_postproc_wf``)

    Inputs
    ------
    fmap : :obj:`os.PathLike`
        Fully preprocessed :math:`B_0` field nonuniformity map (aka *fieldmap*).
    fmap_ref : :obj:`os.PathLike`
        A preprocessed magnitude/reference image for the fieldmap.
    fmap_mask : :obj:`os.PathLike`
        A brain binary mask corresponding to this fieldmap.

    Outputs
    -------
    out_fmap : :obj:`os.PathLike`
        Postprocessed fieldmap.

    """
    from nipype.interfaces.fsl import SpatialFilter
    from niflow.nipype1.workflows.dmri.fsl.utils import cleanup_edge_pipeline
    workflow = Workflow(name=name)
    inputnode = pe.Node(
        niu.IdentityInterface(fields=["fmap_mask", "fmap_ref", "fmap", "metadata"]),
        name="inputnode",
    )
    outputnode = pe.Node(
        niu.IdentityInterface(fields=["out_fmap", "metadata"]), name="outputnode"
    )
    recenter = pe.Node(
        niu.Function(function=_recenter),
        name="recenter",
        run_without_submitting=True,
    )
    denoise = pe.Node(
        SpatialFilter(
            operation="median",
            kernel_shape="sphere",
            kernel_size=median_kernel_size,
        ),
        name="denoise",
    )
    demean = pe.Node(niu.Function(function=_demean), name="demean")
    cleanup_wf = cleanup_edge_pipeline(name="cleanup_wf")

    # fmt: off
    workflow.connect([
        (inputnode, cleanup_wf, [("fmap_mask", "inputnode.in_mask")]),
        (inputnode, recenter, [(("fmap", _pop), "in_file")]),
        (recenter, denoise, [("out", "in_file")]),
        (denoise, demean, [("out_file", "in_file")]),
        (demean, cleanup_wf, [("out", "inputnode.in_file")]),
        (cleanup_wf, outputnode, [("outputnode.out_file", "out_fmap")]),
        (inputnode, outputnode, [(("metadata", _pop), "metadata")]),
    ])
    # fmt: on
    return workflow


def init_phdiff_wf(omp_nthreads, name="phdiff_wf"):
    r"""
    Generate a :math:`B_0` field from consecutive-phases and phase-difference maps.

    This workflow preprocess phase-difference maps (or generates the phase-difference
    map should two ``phase1``/``phase2`` be provided at the input), and generates
    an image equivalent to BIDS's ``fieldmap`` that can be processed with the
    general fieldmap workflow.

    Besides phase2 - phase1 subtraction, the core of this particular workflow relies
    in the phase-unwrapping with FSL PRELUDE [Jenkinson2003]_.
    Because phase (and phase-difference) maps are clipped in the range
    :math:`[0 \dotsb 2\pi )`, it is necessary to find the integer number of offsets
    that make a region continuously smooth with its neighbors (*phase-unwrapping*).
    FSL PRELUDE takes wrapped maps in the range 0 to 6.28, `as per the user guide
    <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#Step_2_-_Getting_.28wrapped.29_phase_in_radians>`__.

    For the phase-difference maps, recentering back to :math:`[-\pi \dotsb \pi )`
    is necessary.
    After some massaging and with the scaling of the echo separation factor
    :math:`\Delta \text{TE}`, the phase-difference maps are converted into
    an actual :math:`B_0` map in Hz units.
    This implementation derives `originally from Nipype
    <https://github.com/nipy/nipype/blob/0.12.1/nipype/workflows/dmri/fsl/artifacts.py#L514>`__.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from sdcflows.workflows.fit.fieldmap import init_phdiff_wf
            wf = init_phdiff_wf(omp_nthreads=1)

    Parameters
    ----------
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use

    Inputs
    ------
    magnitude : :obj:`os.PathLike`
        A reference magnitude image preprocessed elsewhere.
    phase : :obj:`list` of :obj:`tuple` of (:obj:`os.PathLike`, :obj:`dict`)
        List containing one GRE phase-difference map with its corresponding metadata
        (requires ``EchoTime1`` and ``EchoTime2``), or the phase maps for the two
        subsequent echoes, with their metadata (requires ``EchoTime``).
    mask : :obj:`os.PathLike`
        A brain mask calculated from the magnitude image.

    Outputs
    -------
    fieldmap : :obj:`os.PathLike`
        The estimated fieldmap in Hz.  # TODO: write metadata "Units"

    References
    ----------
    .. [Jenkinson2003] Jenkinson, M. (2003) Fast, automated, N-dimensional phase-unwrapping
        algorithm. MRM 49(1):193-197. doi:`10.1002/mrm.10354 <10.1002/mrm.10354>`__.

    """
    from nipype.interfaces.fsl import PRELUDE
    from ...interfaces.fmap import Phasediff2Fieldmap, PhaseMap2rads, SubtractPhases

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
The corresponding phase-map(s) were phase-unwrapped with `prelude` (FSL {PRELUDE.version}).
"""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=["magnitude", "phase", "mask"]), name="inputnode"
    )

    outputnode = pe.Node(
        niu.IdentityInterface(fields=["fieldmap"]), name="outputnode",
    )

    def _split(phase):
        return phase

    split = pe.MapNode(  # We cannot use an inline connection function with MapNode
        niu.Function(function=_split, output_names=["map_file", "meta"]),
        iterfield=["phase"],
        run_without_submitting=True,
        name="split",
    )

    # phase diff -> radians
    phmap2rads = pe.MapNode(
        PhaseMap2rads(),
        name="phmap2rads",
        iterfield=["in_file"],
        run_without_submitting=True,
    )
    # FSL PRELUDE will perform phase-unwrapping
    prelude = pe.Node(PRELUDE(), name="prelude")

    calc_phdiff = pe.Node(
        SubtractPhases(), name="calc_phdiff", run_without_submitting=True
    )
    compfmap = pe.Node(Phasediff2Fieldmap(), name="compfmap")

    # fmt: off
    workflow.connect([
        (inputnode, split, [("phase", "phase")]),
        (inputnode, prelude, [("magnitude", "magnitude_file"),
                              ("mask", "mask_file")]),
        (split, phmap2rads, [("map_file", "in_file")]),
        (phmap2rads, calc_phdiff, [("out_file", "in_phases")]),
        (split, calc_phdiff, [("meta", "in_meta")]),
        (calc_phdiff, prelude, [("phase_diff", "phase_file")]),
        (prelude, compfmap, [("unwrapped_phase_file", "in_file")]),
        (calc_phdiff, compfmap, [("metadata", "metadata")]),
        (compfmap, outputnode, [("out_file", "fieldmap")]),
    ])
    # fmt: on
    return workflow


def _recenter(in_file):
    """Recenter the phase-map distribution to the -pi..pi range."""
    from os import getcwd
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    data = nii.get_fdata(dtype="float32")
    msk = data != 0
    msk[data == 0] = False
    data[msk] -= np.median(data[msk])

    out_file = fname_presuffix(in_file, suffix="_recentered", newpath=getcwd())
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(out_file)
    return out_file


def _demean(in_file, in_mask=None, usemode=True):
    """
    Subtract the median (since it is robuster than the mean) from a map.

    Parameters
    ----------
    usemode : :obj:`bool`
        Use the mode instead of the median (should be even more robust
        against outliers).

    """
    from os import getcwd
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix

    nii = nb.load(in_file)
    data = nii.get_fdata(dtype="float32")

    msk = np.ones_like(data, dtype=bool)
    if in_mask is not None:
        msk[nb.load(in_mask).get_fdata(dtype="float32") < 1e-4] = False

    if usemode:
        from scipy.stats import mode

        data[msk] -= mode(data[msk], axis=None)[0][0]
    else:
        data[msk] -= np.median(data[msk], axis=None)

    out_file = fname_presuffix(in_file, suffix="_demean", newpath=getcwd())
    nb.Nifti1Image(data, nii.affine, nii.header).to_filename(out_file)
    return out_file


def _pop(inlist):
    if isinstance(inlist, (tuple, list)):
        return inlist[0]
    return inlist
