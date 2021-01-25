# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Estimate fieldmaps for :abbr:`SDC (susceptibility distortion correction)`."""
from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as niu
from niworkflows.engine.workflows import LiterateWorkflow as Workflow

LOGGER = logging.getLogger("nipype.workflow")


def init_brainextraction_wf(name="brainextraction_wf"):
    """
    Remove nonbrain tissue from images.

    Parameters
    ----------
    name : :obj:`str`, optional
        Workflow name (default: ``"brainextraction_wf"``)

    Inputs
    ------
    in_file : :obj:`str`
        the GRE magnitude or EPI reference to be brain-extracted

    Outputs
    -------
    out_file : :obj:`str`
        the input file after N4 and smart clipping
    out_brain : :obj:`str`
        the output file, just the brain extracted
    out_mask : :obj:`str`
        the calculated mask
    out_probseg : :obj:`str`
        a probability map that the random walker reached
        a given voxel (some sort of "soft" brainmask)

    """
    from nipype.interfaces.ants import N4BiasFieldCorrection
    from ..interfaces.brainmask import BrainExtraction
    from ..interfaces.utils import IntensityClip

    wf = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(fields=("in_file",)), name="inputnode")
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=(
                "out_file",
                "out_brain",
                "out_mask",
                "out_probseg",
            )
        ),
        name="outputnode",
    )
    clipper_pre = pe.Node(IntensityClip(), name="clipper_pre")

    # de-gradient the fields ("bias/illumination artifact")
    n4 = pe.Node(
        N4BiasFieldCorrection(
            dimension=3,
            copy_header=True,
            n_iterations=[50] * 5,
            convergence_threshold=1e-7,
            shrink_factor=4,
        ),
        n_procs=8,
        name="n4",
    )
    clipper_post = pe.Node(IntensityClip(p_max=100.0), name="clipper_post")
    masker = pe.Node(BrainExtraction(), name="masker")

    # fmt:off
    wf.connect([
        (inputnode, clipper_pre, [("in_file", "in_file")]),
        (clipper_pre, n4, [("out_file", "input_image")]),
        (n4, clipper_post, [("output_image", "in_file")]),
        (clipper_post, masker, [("out_file", "in_file")]),
        (clipper_post, outputnode, [("out_file", "out_file")]),
        (masker, outputnode, [("out_file", "out_brain"),
                              ("out_mask", "out_mask"),
                              ("out_probseg", "out_probseg")]),
    ])
    # fmt:on

    return wf
