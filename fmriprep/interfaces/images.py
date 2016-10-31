from shutil import copy

from nipype import logging
from nipype.interfaces.base import (
    traits, isdefined, TraitedSpec, BaseInterface, BaseInterfaceInputSpec, 
    File, InputMultiPath, OutputMultiPath, traits
)

from fmriprep.interfaces.bids import _splitext

class ImageDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = traits.Directory(
        desc='Path to the base directory for storing data.')
    in_file = InputMultiPath(File(exists=True), mandatory=True,
                             desc='the image to be saved')
    base_file = traits.Str(desc='the input func file')
    overlay_file = traits.Str(desc='the input func file')
    origin_file = File(
        exists=True,
        mandatory=False,
        desc='File from the dataset that image is primarily derived from'
    )

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
        fname, _ = _splitext(self.inputs.origin_file)
        _, ext = _splitext(self.inputs.in_file[0])

        m = re.search(
            '^(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<ses_id>ses-[a-zA-Z0-9]+))?'
            '(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
            '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?',
            fname
        )

        base_directory = os.getcwd()
        if isdefined(self.inputs.base_directory):
            base_directory = op.abspath(self.inputs.base_directory)

        make_folder(out_path)

        base_fname = op.join(out_path, fname)

        formatstr = '{bname}_{suffix}{ext}'
        if len(self.inputs.in_file) > 1:
            formatstr = '{bname}_{suffix}{i:04d}{ext}'

        for i, fname in enumerate(self.inputs.in_file):
            out_file = formatstr.format(
                bname=base_fname,
                suffix=self.inputs.suffix,
                i=i,
                ext=ext)
            self._results['out_file'].append(out_file)
            copy(self.inputs.in_file[i], out_file)

        return runtime

    def _list_outputs(self):
        return self._results
