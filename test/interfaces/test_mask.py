# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
''' Testing module for fmriprep.interfaces.mask '''
import unittest
import mock
import nibabel as nb
import numpy as np

from fmriprep.interfaces.mask import BinarizeSegmentation

class TestMask(unittest.TestCase):
    ''' Testing class for fmriprep.interfaces.mask '''

    segmentation_nii = nb.Nifti1Image(np.array([[[0, 1], [2, 3]],
                                                [[3, 2], [1, 0]]]), np.eye(4))

    @mock.patch.object(nb, 'load', return_value=segmentation_nii)
    @mock.patch.object(nb.nifti1, 'save')
    @mock.patch.object(nb.nifti1.Nifti1Image, '__eq__', autospec=True,
                       side_effect=lambda me: me.get_data().sum() == 4)
    def test_binarize_segmentation(self, nii_eq, mock_save, mock_load):
        ''' mocked an equality function for niftis.
        it will probably catch errors but not guaranteed '''
        # set up
        segmentation = 'README.rst' # convenient existing file
        out_file = 'setup.py'

        # run
        BinarizeSegmentation(in_segments=segmentation, false_values=[0, 1], out_mask=out_file).run()

        # assert
        dummy_mask = nb.Nifti1Image(np.array([]), np.eye(4))

        mock_load.assert_called_once_with(segmentation)
        mock_save.assert_called_once_with(dummy_mask, out_file)
