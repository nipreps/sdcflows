# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Preprocessing of consecutive-phases and phase-difference maps for :math:`B_0` estimation."""

from nipype.interfaces import utility as niu
from nipype.pipeline import engine as pe
from niworkflows.engine.workflows import LiterateWorkflow as Workflow


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

            from sdcflows.workflows.phdiff import init_phdiff_wf
            wf = init_phdiff_wf(omp_nthreads=1)

    Parameters
    ----------
    omp_nthreads : int
        Maximum number of threads an individual process may use

    Inputs
    ------
    magnitude : :obj:`os.pathlike`
        A reference magnitude image preprocessed elsewhere.
    phase : list of tuple(os.pathlike, dict)
        List containing one GRE phase-difference map with its corresponding metadata
        (requires ``EchoTime1`` and ``EchoTime2``), or the phase maps for the two
        subsequent echoes, with their metadata (requires ``EchoTime``).
    mask : :obj:`os.pathlike`
        A brain mask calculated from the magnitude image.

    Outputs
    -------
    fieldmap : :obj:`os.pathlike`
        The estimated fieldmap in Hz.  # TODO: write metadata "Units"

    References
    ----------
    .. [Jenkinson2003] Jenkinson, M. (2003) Fast, automated, N-dimensional phase-unwrapping
        algorithm. MRM 49(1):193-197. doi:`10.1002/mrm.10354 <10.1002/mrm.10354>`__.

    """
    from nipype.interfaces.fsl import PRELUDE
    from ..interfaces.fmap import Phasediff2Fieldmap, PhaseMap2rads, SubtractPhases

    workflow = Workflow(name=name)
    workflow.__desc__ = f"""\
The corresponding phase-map(s) were phase-unwrapped with `prelude` (FSL {PRELUDE.version()}).
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
