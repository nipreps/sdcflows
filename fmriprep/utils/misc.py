from glob import glob
import json
import os

preproc_inputs = {'fieldmaps': [], 'fieldmaps_meta': [], 'epi': '', 'epi_meta': '', 'sbref': '',
                  'sbref_meta': '', 't1': ''}

def _walk_dir_for_prefix(target_dir, prefix):
    return [x for x in next(os.walk(target_dir))[1]
            if x.startswith(prefix)]

def collect_bids_data(dataset, include_types=None):
    imaging_data = {}
    if include_types is None:
        # include all scan types by default
        include_types = ['func', 'anat', 'fmap', 'dwi']

    subjects = _walk_dir_for_prefix(dataset, 'sub-')
    if len(subjects) == 0:
        raise GeneratorExit("No BIDS subjects found to examine.")

    for subject in subjects:
        if subject not in imaging_data:
            imaging_data[subject] = {}
        subj_dir = os.path.join(dataset, subject)

        sessions = _walk_dir_for_prefix(subj_dir, 'ses-')

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
                    imaging_data[subject][session] = preproc_inputs.copy()
                scan_files = glob(os.path.join(
                    dataset, session, scan_type,
                    '*'))

                for scan_file in scan_files:
                    filename = scan_file.split('/')[-1]
                    modality = filename.split('_')[-1]
                    if 'bold.nii' in modality:
                        imaging_data[subject][session]['epi'] = scan_file
                    elif 'bold.json' in modality:
                        fp = open(scan_file)
                        scan_file_json = json.load(fp)
                        fp.close()
                        imaging_data[subject][session]['epi_meta'] = scan_file_json
                    elif 'sbref.nii' in modality:
                        imaging_data[subject][session]['sbref'] = scan_file
                    elif 'sbref.json' in modality:
                        fp = open(scan_file)
                        scan_file_json = json.load(fp)
                        fp.close()
                        imaging_data[subject][session]['sbref_meta'] = scan_file_json
                    elif 'T1W' in modality:
                        imaging_data[subject][session]['t1'] = scan_file
                    elif 'epi.nii' in modality:
                        imaging_data[subject][session]['fieldmaps'].append(scan_file)
                    elif 'epi.json' in modality:
                        fp = open(scan_file)
                        scan_file_json = json.load(fp)
                        fp.close()
                        imaging_data[subject][session]['fieldmaps'].append(scan_file_json)
                    else:
                        pass
    return imaging_data

# id we know the subject id and session name we can collect the data files for 
# just that subject/sesssion
def collect_sub_ses_data(dataset, subject, session):
    imaging_data = preproc_inputs.copy()
    if include_types is None:
        # include all scan types by default
        include_types = ['func', 'anat', 'fmap', 'dwi']

    subjects = _walk_dir_for_prefix(dataset, subject)
    if len(subjects) == 0:
        raise GeneratorExit("No BIDS subjects found to examine.")

    for subject in subjects:
        subj_dir = os.path.join(dataset, subject)

        sessions = _walk_dir_for_prefix(subj_dir, session)

        for scan_type in include_types:
            # seems easier to consider the case of multi-session vs.
            # single session separately?
            if len(sessions) > 0:
                subject_sessions = [os.path.join(subject, x)
                                    for x in sessions]
            else:
                subject_sessions = [subject]

            for session in subject_sessions:
                scan_files = glob(os.path.join(
                    dataset, session, scan_type,
                    '*'))

                for scan_file in scan_files:
                    filename = scan_file.split('/')[-1]
                    modality = filename.split('_')[-1]
                    if 'bold.nii' in modality:
                        imaging_data['epi'] = scan_file
                    elif 'bold.json' in modality:
                        fp = open(scan_file)
                        scan_file_json = json.load(fp)
                        fp.close()
                        imaging_data['epi_meta'] = scan_file_json
                    elif 'sbref.nii' in modality:
                        imaging_data['sbref'] = scan_file
                    elif 'sbref_json' in modality:
                        fp = open(scan_file)
                        scan_file_json = json.load(fp)
                        fp.close()
                        imaging_data['sbref_meta'] = scan_file_json
                    elif 'T1W' in modality:
                        imaging_data['t1'] = scan_file
                    elif 'epi.nii' in modality:
                        imaging_data['fieldmaps'].append(scan_file)
                    elif 'epi.json' in modality:
                        fp = open(scan_file)
                        scan_file_json = json.load(fp)
                        fp.close()
                        imaging_data['fieldmaps'].append(scan_file_json)
                    else:
                        pass
    return imaging_data


if __name__ == '__main__':
    pass
