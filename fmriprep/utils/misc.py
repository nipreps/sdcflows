import copy
from glob import glob
import itertools
import json
import os
import pkg_resources as pkgr
import pprint
import re

from grabbit import Layout

INPUTS_SPEC = {'fieldmaps': [], 'epi': [], 'sbref': [], 't1': ''}

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
    'phasediff': r"phasediff\.nii",
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
    
    ret_list = []

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
        'fieldmaps': {'fieldmap': '.*'},
        'epi': {'func': '.*'},
        'sbref': {'type': 'sbref'},
        't1': {'type': 'T1w'}
    }
    print(session_list)
    print(run_list)
    for sess in session_list:
        sess_run_kwargs = {}
        sess_run_kwargs['subject'] = subject
        if sess:
            sess_run_kwargs['session'] = sess
        for run in run_list:
            if run:
                sess_run_kwargs['run'] = run
            imaging_data = copy.deepcopy(INPUTS_SPEC)
            for key in queries.keys():
                layout_kwargs = dict(sess_run_kwargs, **queries[key])
                print(layout_kwargs)
                files = [x.filename for x in layout.get(**layout_kwargs)]
                if len(files) == 0 and layout_kwargs.pop('run', None):
                    files = [x for x in layout.get(**layout_kwargs)]
                    run_found = False
                    for file in files:
                        if hasattr(file, 'run'):
                            run_found = True
                    if not run_found:
                        files = [x.filename for x in files]
                    else:
                        files = []
                imaging_data[key] = files
            ret_list.append(imaging_data)
    return ret_list
            

if __name__ == '__main__':
    pass
