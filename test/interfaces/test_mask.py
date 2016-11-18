# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Testing module for fmriprep.interfaces.mask '''
import os
import unittest
import mock
import logging

import nibabel as nb
import numpy as np

from fmriprep.interfaces.mask import BinarizeSegmentation

logging.disable(logging.INFO)

class TestMask(unittest.TestCase):
    ''' Testing class for fmriprep.interfaces.mask '''

    segmentation_nii = nb.Nifti1Image(np.array([[[0, 1], [2, 3]],
                                                [[3, 2], [1, 0]]]), np.eye(4))

    @mock.patch.object(nb, 'load', return_value=segmentation_nii)
    @mock.patch.object(nb.nifti1, 'save')
    @mock.patch.object(os.path, 'isfile', return_value=True)
    def test_binarize_segmentation(self, mock_file_exists, mock_save, mock_load):
        # set up
        segmentation = 'thisfiletotallyexists'
        out_file = 'thisonedoesnot.yet'

        # run
        bi = BinarizeSegmentation(in_segments=segmentation, false_values=[0, 1], out_mask=out_file)
        bi.run()

        # assert
        mock_load.assert_called_once_with(segmentation)

        mask, filename = mock_save.call_args[0]

        out_file_abs = os.path.abspath(out_file)
        self.assertEqual(filename, out_file_abs)
        self.assertEqual(bi.aggregate_outputs().get()['out_mask'], out_file_abs)

        self.assertTrue(np.allclose(mask.get_data(), np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]]),
                                    atol=0, rtol=0)) # all elements of each array are equal
