#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-06-03 09:35:13
# @Last Modified by:   oesteban
# @Last Modified time: 2016-08-17 15:32:43
from lockfile import LockFile
import os
import os.path as op
import pkg_resources as pkgr
import re
from shutil import copy
import simplejson as json

from nipype import logging
from nipype.interfaces.base import (traits, isdefined, TraitedSpec, BaseInterface,
                                    BaseInterfaceInputSpec, File, InputMultiPath, OutputMultiPath)

from fmriprep.utils.misc import collect_bids_data

LOGGER = logging.getLogger('interface')

class FileNotFoundError(IOError):
    pass


class BIDSDataGrabberInputSpec(BaseInterfaceInputSpec):
    subject_data = traits.DictStrAny()
    subject_id = traits.Str()


class BIDSDataGrabberOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='output data structure')
    fmap = OutputMultiPath(desc='output fieldmaps')
    func = OutputMultiPath(desc='output functional images')
    sbref = OutputMultiPath(desc='output sbrefs')
    t1w = OutputMultiPath(desc='output T1w images')


class BIDSDataGrabber(BaseInterface):
    input_spec = BIDSDataGrabberInputSpec
    output_spec = BIDSDataGrabberOutputSpec

    def __init__(self, **inputs):
        self._results = {'out_dict': {}}
        super(BIDSDataGrabber, self).__init__(**inputs)

    def _run_interface(self, runtime):
        bids_dict = self.inputs.subject_data

        self._results['out_dict'] = bids_dict

        self._results['t1w'] = bids_dict['t1w']
        if not bids_dict['t1w']:
            raise FileNotFoundError('No T1w images found for subject sub-{}'.format(
                self.inputs.subject_id))

        self._results['func'] = bids_dict['func']
        if not bids_dict['func']:
            raise FileNotFoundError('No functional images found for subject sub-{}'.format(
                self.inputs.subject_id))

        for imtype in ['fmap', 'sbref']:
            self._results[imtype] = bids_dict[imtype]
            if not bids_dict[imtype]:
                LOGGER.warn('No \'{}\' images found for sub-{}'.format(
                    imtype, self.inputs.subject_id))


        return runtime

    def _list_outputs(self):
        return self._results


class DerivativesDataSinkInputSpec(BaseInterfaceInputSpec):
    base_directory = traits.Directory(
        desc='Path to the base directory for storing data.')
    in_file = InputMultiPath(File(), exists=True, mandatory=True,
                             desc='the object to be saved')
    source_file = File(exists=True, mandatory=True, desc='the input func file')
    suffix = traits.Str('', mandatory=True, desc='suffix appended to source_file')

class DerivativesDataSinkOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True, desc='written file path'))

class DerivativesDataSink(BaseInterface):
    input_spec = DerivativesDataSinkInputSpec
    output_spec = DerivativesDataSinkOutputSpec
    _always_run = True

    def __init__(self, **inputs):
        self._results = {'out_file': []}
        super(DerivativesDataSink, self).__init__(**inputs)

    def _run_interface(self, runtime):
        fname, _ = _splitext(self.inputs.source_file)
        _, ext = _splitext(self.inputs.in_file[0])

        m = re.search(
            '^(?P<subject_id>sub-[a-zA-Z0-9]+)(_(?P<ses_id>ses-[a-zA-Z0-9]+))?'
            '(_(?P<task_id>task-[a-zA-Z0-9]+))?(_(?P<acq_id>acq-[a-zA-Z0-9]+))?'
            '(_(?P<rec_id>rec-[a-zA-Z0-9]+))?(_(?P<run_id>run-[a-zA-Z0-9]+))?',
            fname
        )

        # TODO this quick and dirty modality detection needs to be implemented
        # correctly
        mod = 'func'
        if 'anat' in op.dirname(self.inputs.source_file):
            mod = 'anat'
        elif 'dwi' in op.dirname(self.inputs.source_file):
            mod = 'dwi'

        base_directory = os.getcwd()
        if isdefined(self.inputs.base_directory):
            base_directory = op.abspath(self.inputs.base_directory)

        out_path = 'derivatives/{subject_id}'.format(**m.groupdict())
        if m.groupdict().get('ses_id') is not None:
            out_path += '/{ses_id}'.format(**m.groupdict())
        out_path += '/{}'.format(mod)

        out_path = op.join(base_directory, out_path)
        with LockFile(op.join(base_directory, '.fmriprep-lock')):
            if not op.exists(out_path):
                os.makedirs(out_path)

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


class ReadSidecarJSONInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='the input nifti file')
    fields = traits.List(traits.Str, desc='get only certain fields')


class ReadSidecarJSONOutputSpec(TraitedSpec):
    out_dict = traits.Dict()


class ReadSidecarJSON(BaseInterface):
    """
    An utility to find and read JSON sidecar files of a BIDS tree
    """

    input_spec = ReadSidecarJSONInputSpec
    output_spec = ReadSidecarJSONOutputSpec

    def __init__(self, **inputs):
        self._results = {}
        super(ReadSidecarJSON, self).__init__(**inputs)

    def _run_interface(self, runtime):
        metadata = get_metadata_for_nifti(self.inputs.in_file)

        if isdefined(self.inputs.fields) and self.inputs.fields:
            for fname in self.inputs.fields:
                self._results[fname] = metadata[fname]
        else:
            self._results = metadata

        return runtime

    def _list_outputs(self):
        out = self.output_spec().get()
        out['out_dict'] = self._results
        return out

def get_metadata_for_nifti(in_file):
    """Fetchs metadata for a given nifi file"""
    in_file = op.abspath(in_file)

    fname, ext = op.splitext(in_file)
    if ext == '.gz':
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext

    side_json = fname + '.json'
    fname_comps = side_json.split('/')[-1].split("_")

    session_comp_list = []
    subject_comp_list = []
    top_comp_list = []
    ses = None
    sub = None

    for comp in fname_comps:
        if comp[:3] != "run":
            session_comp_list.append(comp)
            if comp[:3] == "ses":
                ses = comp
            else:
                subject_comp_list.append(comp)
                if comp[:3] == "sub":
                    sub = comp
                else:
                    top_comp_list.append(comp)


    top_json = "/" + "_".join(top_comp_list)
    potential_json = [top_json]

    subject_json = "/" + sub + "/" + "_".join(subject_comp_list)
    potential_json.append(subject_json)

    if ses:
        session_json = "/" + sub + "/" + ses + "/" + "_".join(session_comp_list)
        potential_json.append(session_json)

    potential_json.append(side_json)

    merged_param_dict = {}
    for json_file_path in potential_json:
        if op.isfile(json_file_path):
            with open(json_file_path, 'r') as jsonfile:
                param_dict = json.load(jsonfile)
                merged_param_dict.update(param_dict)

    return merged_param_dict

def _splitext(fname):
    fname, ext = op.splitext(op.basename(fname))
    if ext == '.gz':
        fname, ext2 = op.splitext(fname)
        ext = ext2 + ext
    return fname, ext
