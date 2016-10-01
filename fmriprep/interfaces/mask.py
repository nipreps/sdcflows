''' Utility functions related to masks and masking '''
#!/usr/bin/env python
import numpy as np
import nibabel as nb
from nipype.interfaces.base import (traits, TraitedSpec, BaseInterface,
                                    BaseInterfaceInputSpec, File)

class BinarizeSegmentationInputSpec(BaseInterfaceInputSpec):
    in_segments = File(exists=True, mandatory=True, desc='3d tissue class segmentation. '
                       'Values are integers')
    false_values = traits.List([0], usedefault=True,
                               desc='list of values in in_mask that are to be set to false (0). '
                               'All others will be set to true (1)')
    out_mask = File('mask.nii', exists=False, usedefault=True,
                    desc='the file name to save the output to')

class BinarizeSegmentationOutputSpec(TraitedSpec):
    ''' out_mask defaults to 'mask.nii' (see input spec) '''
    out_mask = File(exists=True, desc='binarized_mask')

class BinarizeSegmentation(BaseInterface):
    '''
    Utility for turning a segmentation with integer values into a binary map.
    An example input is the segmentation output by fsl.FAST

    Use case: Get a white matter mask
    >>> biseg = BinarizeSegmentation()
    >>> biseg.inputs.in_segments = 'fast_out.nii'
    >>> biseg.inputs.false_values = [0, 1, 2]
    >>> biseg.run()

    Use case: Get white matter and CSF (as for aCompCor)
    >>> BinarizeSegmentation(in_seg='fast_out.nii', false_values=[0, 1]).run() # check
    '''
    input_spec = BinarizeSegmentationInputSpec
    output_spec = BinarizeSegmentationOutputSpec

    def __init__(self, **inputs):
        super(BinarizeSegmentation, self).__init__(**inputs)

    def _run_interface(self, runtime):
        segments_data, segments_affine = self._get_inputs()

        mapper = np.vectorize(lambda orig_val: orig_val not in self.inputs.false_values)
        bimap = mapper(segments_data)

        bimap_nii = nb.Nifti1Image(bimap.astype(int), segments_affine)
        nb.nifti1.save(bimap_nii, self.inputs.out_mask)

        return runtime

    def _get_inputs(self):
        ''' manipulates inputs into useful form. does preliminary input-checking '''
        segments_nii = nb.load(self.inputs.in_segments)
        segments_data = segments_nii.get_data()
        segments_affine = segments_nii.affine

        if str(segments_data.dtype)[:2] == 'int':
            raise ValueError('Segmentation must have integer values. Input {} had {}s'
                             .format(self.inputs.in_segments, segments_data.dtype))
        if segments_data.ndim != 3:
            raise ValueError('Segmentation must be 3-D. Input {} has shape {}'
                             .format(self.inputs.in_segments, segments_data.shape))

        return segments_data, segments_affine

    def _list_outputs(self):
        return {'out_mask': self.inputs.out_mask}
