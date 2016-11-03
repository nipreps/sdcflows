import json
import re
import os
import os.path as op
from shutil import copy

from nipype import logging
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, 
    File, InputMultiPath, OutputMultiPath, traits
)

from fmriprep.interfaces.bids import _splitext 
from fmriprep.utils.misc import make_folder

class ImageDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = traits.Directory(
        desc='Path to the base directory for storing data.')
    in_file = traits.Str(desc='the image to be saved')
    base_file = traits.Str(desc='the input func file')
    overlay_file = traits.Str(desc='the input func file')
    origin_file = traits.Str(desc='File from the dataset that image is primarily derived from')

class ImageDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True, desc='written file path'))

class ImageDataSink(BaseInterface):
    input_spec = ImageDataSinkInputSpec
    output_spec = ImageDataSinkOutputSpec
    _always_run = True

    def __init__(self, **inputs):
        self._results = {'out_file': []}
        super(ImageDataSink, self).__init__(**inputs)

    def _run_interface(self, runtime):
        origin_fname, _ = _splitext(self.inputs.origin_file)

        image_inputs = {}
        if isdefined(self.inputs.base_file):
            image_inputs['base_file'] = self.inputs.base_file
        if isdefined(self.inputs.overlay_file):
            image_inputs['overlay_file'] = self.inputs.overlay_file
        if isdefined(self.inputs.origin_file):
            image_inputs['origin_file'] = self.inputs.overlay_file

        m = re.search(
            '^(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<ses_id>ses-[a-zA-Z0-9]+))?'
            '(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
            '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?',
            origin_fname
        )

        base_directory = os.getcwd()
        if isdefined(self.inputs.base_directory):
            base_directory = op.abspath(self.inputs.base_directory)

        out_path = 'images/{subject_id}'.format(**m.groupdict())

        out_path = op.join(base_directory, out_path)

        make_folder(out_path)

        _, out_filename = op.split(self.inputs.in_file)

        #  test incoming origin file for these identifiers, if they exist
        #  we want to fold them into out filename
        group_keys = ['ses_id', 'task_id', 'acq_id', 'rec_id', 'run_id']
        if [x for x in group_keys if m.groupdict().get(x)]:
            out_filename, ext = _splitext(out_filename)
            out_filename = '{}_{}.{}'.format(out_filename, origin_fname, ext)

        out_file = op.join(out_path, out_filename)


        self._results['out_file'].append(out_file)
        copy(self.inputs.in_file, out_file)
        json_fname, _ = _splitext(out_filename)

        json_out_filename = '{}.{}'.format(json_fname, 'json')
        json_out_file = op.join(out_path, json_out_filename)
        with open(json_out_file, 'w') as fp:
            json.dump(image_inputs, fp)

        return runtime

    def _list_outputs(self):
        return self._results
