''' Testing module for fmriprep.workflows.base '''
import pytest

import numpy as np
from nilearn.image import load_img
from ..utils import init_bold_reference_wf


def symmetric_overlap(img1, img2):
    mask1 = load_img(img1).get_data() > 0
    mask2 = load_img(img2).get_data() > 0

    total1 = np.sum(mask1)
    total2 = np.sum(mask2)
    overlap = np.sum(mask1 & mask2)
    return overlap / np.sqrt(total1 * total2)


@pytest.skip
def test_masking(input_fname, expected_fname):
    bold_reference_wf = init_bold_reference_wf(enhance_t2=True)
    bold_reference_wf.inputs.inputnode.bold_file = input_fname
    res = bold_reference_wf.run()

    combine_masks = [node for node in res.nodes if node.name.endswith('combine_masks')][0]
    overlap = symmetric_overlap(expected_fname,
                                combine_masks.result.outputs.out_file)

    assert overlap < 0.95, input_fname
