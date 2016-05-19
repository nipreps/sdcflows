from glob import glob
import copy
import json
import os
import re

INPUTS_SPEC = {'fieldmaps': [], 'fieldmaps_meta': [], 'epi': '',
               'epi_meta': '', 'sbref': '', 'sbref_meta': '', 't1': ''}


def _walk_dir_for_prefix(target_dir, prefix):
    return [x for x in next(os.walk(target_dir))[1]
            if x.startswith(prefix)]


def get_subject(bids_root, subject_id, session_id=None, run_id=None,
                include_types=None):
    """
    Returns the imaging_data structure for the subject subject_id.
    If session is None, then the BIDS structure is not multisession.
    If run_id is None, it is assumed that the session does not have several
    runs.
    """
    if include_types is None:
        # Please notice that dwi is not here
        include_types = ['func', 'anat', 'fmap']
    subject_data = collect_bids_data(bids_root, include_types=None)
    subject_data = subject_data['sub-' + subject_id]

    if session_id is None:
        subject_data = subject_data[list(subject_data.keys())[0]]
    else:
        raise NotImplementedError

    if run_id is not None:
        raise NotImplementedError

    return subject_data


# if no scan_subject or scan_session are defined return all bids data for a
# given bids directory. Otherwise just the data for a given subject or scan
# can be returned
def collect_bids_data(dataset, include_types=None, scan_subject='sub-',
                      scan_session='ses-'):
    imaging_data = {}
    if include_types is None:
        include_types = ['func', 'anat', 'fmap', 'dwi']

    subjects = _walk_dir_for_prefix(dataset, scan_subject)
    if len(subjects) == 0:
        raise GeneratorExit("No BIDS subjects found to examine.")

    for subject in subjects:
        if subject not in imaging_data:
            imaging_data[subject] = {}
        subj_dir = os.path.join(dataset, subject)

        sessions = _walk_dir_for_prefix(subj_dir, scan_session)

        for scan_type in include_types:
            # seems easier to consider the case of multi-session vs.
            # single session separately?
            if len(sessions) > 0:
                subject_sessions = [os.path.join(subject, x)
                                    for x in sessions]
            else:
                subject_sessions = [subject]

            for session in subject_sessions:
                if session not in imaging_data[subject]:
                    imaging_data[subject][session] = copy.deepcopy(INPUTS_SPEC)
                scan_files = glob(os.path.join(dataset, session, scan_type,
                                               '*'))

                for scan_file in scan_files:
                    filename = scan_file.split('/')[-1]
                    filename_parts = filename.split('_')
                    modality = filename_parts[-1]
                    # temporary conditional until runs and tasks are handled
                    # in the imaging data structure
                    if 'rest_acq-LR_run-1' in filename:
                        if 'bold.nii' in modality:
                            imaging_data[subject][session]['epi'] = scan_file
                        elif 'bold.json' in modality:
                            imaging_data[subject][session]['epi_meta'] = scan_file
                        elif 'sbref.nii' in modality:
                            imaging_data[subject][session]['sbref'] = scan_file
                        elif 'sbref.json' in modality:
                            imaging_data[subject][session]['sbref_meta'] = scan_file
                        else:
                            pass
                    else:
                        if 'epi.nii' in modality:
                            imaging_data[subject][session]['fieldmaps'].append(scan_file)
                        elif 'epi.json' in modality:
                            imaging_data[subject][session]['fieldmaps_meta'].append(scan_file)
                        elif 'T1w.nii' in modality:
                            imaging_data[subject][session]['t1'] = scan_file
                        else:
                            pass
    return imaging_data


class SubjectIDNotFound(Exception):
    pass


class SessionIDNotFound(Exception):
    pass


class RunIDNotFound(Exception):
    pass


class AcqIDNotFound(Exception):
    pass


class RecIDNotFound(Exception):
    pass


class TaskIDNotFound(Exception):
    pass


def get_atomic_data(dataset, include_types=None, sub_id=None, 
                          ses_id=None, run_id=None, acq_id=None, rec_id=None, 
                          task_id=None):
    imaging_data = {}
    if include_types is None:
        include_types = ['func', 'anat', 'fmap', 'dwi']
    file_lists = {}
    for include_type in include_types:
        file_lists[include_type] = []

    bids_files = []
    for path, subdirs, files in os.walk('.'):
        for name in files:
            bids_files.append(os.path.join(path, name))

    imaging_data = copy.deepcopy(INPUTS_SPEC)
    for bids_file in bids_files:
        filename = bids_file.split('/')[-1]
        filename_parts = filename.split('_')
        modality = filename_parts[-1]

        file_type = None
        for include_type in include_types:
            type_match = re.search('/{}/'.format(include_type), bids_file)
            if type_match is not None:
                file_type = include_type

        if sub_id is not None:
            subject_match = re.search('/sub-{})/'.format(sub_id), bids_file)
            if not subject_match:
                raise SubjectIDNotFound

        if ses_id is not None:
            session_match = re.search('/ses-{}/'.format(ses_id), bids_file)
            if not session_match:
                raise SessionIDNotFound

        #  Fmap covered
        if file_type == 'fmap':
            if 'epi.nii' in modality:
                imaging_data['fieldmaps'].append(bids_file)
            elif 'epi.json' in modality:
                imaging_data['fieldmaps_meta'].append(bids_file)
            continue

        if (run_id is not None) and (file_type != 'fmap'):
            run_match = re.search('run-{}_'.format(run_id), bids_file)
            if not run_match:
                raise RunIDNotFound

        if acq_id is not None:
            acq_match = re.search('acq-{}_'.format(acq_id), bids_file)
            if not acq_match:
                raise AcqIDNotFound

        #  DWI covered
        if file_type == 'dwi':
            if 'sbref.nii' in modality:
                imaging_data[subject][session]['sbref'] = bids_file
            elif 'sbref.json' in modality:
                imaging_data[subject][session]['sbref_meta'] = bids_file
            continue

        if rec_id is not None:
            rec_match = re.search('rec-{}_'.format(rec_id), bids_file)
            if not rec_match:
                raise RecIDNotFound

        #  Anat covered
        if file_type == 'anat':
            if 'T1w.nii' in modality:
                imaging_data['t1'] = bids_file
            continue

        if task_id is not None:
            task_match = re.search('task-{}_'.format(task_id), bids_file)
            if not task_match:
                raise TaskIDNotFound
        #  Func covered
        if file_type == 'func':
            if 'bold.nii' in modality:
                imaging_data[subject][session]['epi'] = bids_file
            elif 'bold.json' in modality:
                imaging_data[subject][session]['epi_meta'] = bids_file
            continue

    return imaging_data

#  get all functional images for a subject/session. Once we have these we can
#  look at the other modalities and see which ones should be associated with
#  the images based on their file names
def get_funcs(func_files):
    functional_data = []
    for func_file in func_files:
        filename = func_file.split('/')[-1]
        filename_parts = filename.split('_')
        modality = filename_parts[-1]
        if 'bold.nii' not in modality:
            continue
        
        bold = func_file
        bold_json = re.sub('nii(\.gz)?$', 'json', bold)
        functional_data.append((bold, bold_json))
   
    return functional_data

#  Given any run, acq, or rec ids we expect to find a t1 file with them in
#  their filename. If none of theses are present there should only be one
#  suitable file to be used by all functional images of the same subject and
#  session.
def get_anat(anat_files, run_id=None, acq_id=None, rec_id=None):
    t1 = []
    for anat_file in anat_files:    
        if run_id is not None:
            run_match = re.search('run-{}_'.format(run_id), anat_file)
            if not run_match:
                continue

        if acq_id is not None:
            acq_match = re.search('acq-{}_'.format(acq_id), anat_file)
            if not acq_match:
                continue

        if rec_id is not None:
            rec_match = re.search('rec-{}_'.format(rec_id), anat_file)
            if not rec_match:
                continue

        if file_type == 'anat':
            if 'T1w.nii' in modality:
                t1.append(anat_file)
            continue

    if len(t1) == 0:
        raise NoT1sFound
    elif len(t1) > 1:
        raise MultipleValidT1s
    else:
        return t1[0]
        
def get_dwi(dwi_files, run_id=None, acq_id=None):
    dwi_data = []
    for dwi_file in dwi_files:    
        if run_id is not None:
            run_match = re.search('run-{}_'.format(run_id), dwi_file)
            if not run_match:
                continue

        if acq_id is not None:
            acq_match = re.search('acq-{}_'.format(acq_id), dwi_file)
            if not acq_match:
                continue

        if 'sbref.nii' in modality:
            sbref = func_file
            bold_json = re.sub('nii(\.gz)?$', 'json', bold)
            dwi_data.append((bold, bold_json))

    if len(dwi_data) == 0:
        raise NoDWIsFound
    elif len(dwi_data) > 1:
        raise MultipleValidDWIs
    else:
        return dwi_data[0]

def get_fmap(fmap_files, run_id=None, acq_id=None):
    return

def check_intended_for(fmap_json_file, func_file):
    try:
        fmap_json_fp = open(fmap_json_file, 'r')
        fmap_json = json.loads(fmap_json_fp.readlines())
        intended_for = fmap_json["IntendedFor"]
    except OSError:
        return None
    except ValueError:
        raise InvalidFmapJsonFile
    except KeyError:
        return None
    

def get_ids(dataset):
    ids = {'subjects': set(), 'sessions': set(), 'tasks': set(), 
           'acquisitions': set(), 'reconstructions': set()}

    bids_files = []
    for path, subdirs, files in os.walk(dataset):
        for name in files:
            bids_files.append(os.path.join(path, name))

    for bids_file in bids_files:
        subject_match = re.search('(?<=/sub-)(.*?)(?=/)', bids_file)
        if subject_match:
            ids['subjects'].add(subject_match.group())

        session_match = re.search('(?<=/ses-)(.*?)(?=/)', bids_file)
        if session_match:
            ids['sessions'].add(session_match.group())

        run_match = re.search('(?<=run-)(.*?)(?=_)', bids_file)
        if run_match:
            ids['runs'].add(run_match.group())

        acq_match = re.search('(?<=acq-)(.*?)(?=_)', bids_file)
        if acq_match:
            ids['acquisitions'].add(acq_match.group())

        task_match = re.search('(?<=task-)(.*?)(?=_)', bids_file)
        if task_match:
            ids['tasks'].add(task_match.group())

        rec_match = re.search('(?<=rec-)(.*?)(?=_)', bids_file)
        if rec_match:
            ids['reconstructions'].add(rec_match.group())
        
    print(ids)

if __name__ == '__main__':
    get_ids('/Users/rblair/datasets/ds172_R1.0.0')
