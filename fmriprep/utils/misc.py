from glob import glob
import copy
import json
import os
import re

INPUTS_SPEC = {'fieldmaps': [], 'epi': '', 'sbref': [], 't1': ''}


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
                    if 'sbref.nii' in modality:
                        imaging_data[subject][session]['sbref'].append(scan_file)
                    elif 'epi.nii' in modality:
                        imaging_data[subject][session]['fieldmaps'].append(scan_file)
                    elif 'T1w.nii' in modality:
                        imaging_data[subject][session]['t1'] = scan_file
                    # temporary conditional until runs and tasks are handled
                    # in the imaging data structure
                    elif 'rest_acq-LR_run-1' in filename:
                        if 'bold.nii' in modality:
                            imaging_data[subject][session]['epi'] = scan_file
                    else:
                        pass

    return imaging_data

if __name__ == '__main__':
    pass
