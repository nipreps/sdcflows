''' Testing module for fmriprep.workflows.base '''
import pytest

import numpy as np
from nilearn.image import load_img
from ..utils import init_enhance_and_skullstrip_bold_wf


def symmetric_overlap(img1, img2):
    mask1 = load_img(img1).get_data() > 0
    mask2 = load_img(img2).get_data() > 0

    total1 = np.sum(mask1)
    total2 = np.sum(mask2)
    overlap = np.sum(mask1 & mask2)
    return overlap / np.sqrt(total1 * total2)


def test_masking(input_fname, expected_fname):
    enhance_and_skullstrip_bold_wf = init_enhance_and_skullstrip_bold_wf()
    enhance_and_skullstrip_bold_wf.inputs.inputnode.in_file = input_fname
    res = enhance_and_skullstrip_bold_wf.run()

    combine_masks = [node for node in res.nodes if node.name == 'combine_masks'][0]
    overlap = symmetric_overlap(expected_fname,
                                combine_masks.result.outputs.out_file)

    assert overlap < 0.95, input_fname
