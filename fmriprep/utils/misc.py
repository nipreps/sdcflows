import copy
from glob import glob
import itertools
import json
import os
import pkg_resources as pkgr
import re

from grabbit import Layout

INPUTS_SPEC = {'fieldmaps': [], 'func': [], 't1': [], 'sbref': []}

def gen_list(inlist, base=1):
    return range(base, len(inlist) + base)

def _walk_dir_for_prefix(target_dir, prefix):
    return [x for x in next(os.walk(target_dir))[1]
            if x.startswith(prefix)]

def is_fieldmap_file(string):
    is_fieldmap_file = False
    for suffix in fieldmap_suffixes.values():
        if re.search(suffix, string):
            is_fieldmap_file = True
    return is_fieldmap_file

fieldmap_suffixes = {
    'phasediff': r"phasediff[0-9]*\.nii",
    'magnitude': r"magnitude[0-9]*\.nii",
    'phase': r"phase[0-9]+\.nii",
    'fieldmap': r"fieldmap\.nii",
    'topup': r"epi\.nii"
}


def collect_bids_data(dataset, subject, session=None, run=None):
    subject = str(subject)
    if not subject.startswith('sub-'):
        subject = 'sub-{}'.format(subject)

    bids_spec = pkgr.resource_filename('fmriprep', 'data/bids.json')
    layout = Layout(dataset, config=bids_spec)
    
    if session:
        session_list = [session]
    else:
        session_list = layout.unique('session')
        if session_list == []:
            session_list = [None]
    
    if run:
        run_list = [run]
    else:
        run_list = layout.unique('run')
        if run_list == []:
            run_list = [None]

    queries = {
        'fieldmaps': {'fieldmap': '.*', 'ext': 'nii'},
        'epi': {'modality': 'func', 'type': 'bold', 'ext': 'nii'},
        'sbref': {'modality': 'func', 'type': 'sbref', 'ext': 'nii'},
        't1': {'type': 'T1w', 'ext': 'nii'}
    }
    
    #  Add a subject key pair to each query we make so that we only deal with
    #  files related to this workflows specific subject. Could be made opt...
    for key in queries.keys():
        queries[key]['subject'] = subject

    imaging_data = copy.deepcopy(INPUTS_SPEC)
    fieldmap_files = [x.filename for x in layout.get(**queries['fieldmaps'])]
    imaging_data['fieldmaps'] = fieldmap_files
    t1_files = [x.filename for x in layout.get(**queries['t1'])]
    imaging_data['t1'] = t1_files
    sbref_files = [x.filename for x in layout.get(**queries['sbref'])]
    imaging_data['sbref'] = sbref_files

    loop_on = ['session', 'run', 'acquisition', 'task']
    get_kwargs = {}
    
    for key in loop_on:
        unique_list = layout.unique(key)
        if unique_list:
            get_kwargs[key] = unique_list

    query_kwargs = []
    for key in get_kwargs:
        query_kwargs.append([(key, x) for x in get_kwargs[key]])

    query_kwargs = itertools.product(*query_kwargs)

    for elem in query_kwargs:
        epi_files = [x.filename for x 
                     in layout.get(**dict(dict(elem), **queries['epi']))]
        if epi_files:
            imaging_data['func'].append(epi_files)
    return imaging_data


if __name__ == '__main__':
    pass
