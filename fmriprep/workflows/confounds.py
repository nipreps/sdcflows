'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''
from nipype.interfaces import utility, nilearn
from nipype.algorithms import confounds
from nipype.pipeline import engine as pe

def discover_wf(name="ConfoundDiscoverer"):
    ''' All input fields are required.

    Calculates global regressor and tCompCor
        from motion-corrected fMRI ('inputnode.fmri_file').
    Calculates DVARS from the fMRI and an EPI brain mask ('inputnode.epi_mask')
    Calculates frame displacement from MCFLIRT movement parameters ('inputnode.movpar_file')
    Calculates segment regressors and aCompCor
        from the fMRI and a white matter/gray matter/CSF segmentation ('inputnode.t1_seg')

    Saves the confounds in a file ('outputnode.confounds_file')'''

    inputnode = pe.Node(utility.IdentityInterface(fields=['fmri_file', 'movpar_file', 't1_seg',
                                                          'epi_mask']),
                        name='inputnode')
    outputnode = pe.Node(utility.IdentityInterface(fields=['confounds_file']),
                         name='outputnode')

    # Global and segment regressors
    signals = pe.Node(nilearn.SignalExtraction(include_global=True, detrend=True,
                                               class_labels=['white matter',
                                                             'gray matter',
                                                             'CSF']), # check
                      name="SignalExtraction")

    # DVARS
    dvars = pe.Node(confounds.ComputeDVARS(), name="ComputeDVARS", save_all=True,
                    remove_zerovariance=True)

    concat = pe.Node(utility.Function(function=_gather_confounds, input_names=['signals', 'dvars'],
                                      output_names=['combined_out']),
                     name="ConcatConfounds")

    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, signals, [('fmri_file', 'in_file'),
                              ('t1_seg', 'label_files')]),
        (inputnode, dvars, [('fmri_file', 'in_file'),
                            ('epi_mask', 'in_mask')]),

        (signals, concat, [('out_file', 'signals')]),
        (dvars, concat, [('out_all', 'dvars')]),

        (concat, outputnode, [('combined_out', 'confounds_file')])
    ])

    return workflow

def _gather_confounds(signals=None, dvars=None):
    ''' load confounds from the filenames, concatenate together horizontally, and re-save '''
    import pandas as pd

    all_files = [confound for confound in [signals, dvars] if confound != None]

    # make sure there weren't any name conflicts
    if len(all_files) != len(set(all_files)):
        raise RuntimeError('A confound-calculating node over-wrote another confound-calculating'
                           'node\'s results! Check ' + str(all_files))

    confounds = pd.DataFrame()
    for file_name in all_files: # assumes they all have headings already
        new = pd.read_csv(file_name, sep="\t")
        confounds = pd.concat((confounds, new), axis=1)

    combined_out = 'confounds.tsv'
    confounds.to_csv(combined_out, sep="\t")

    return combined_out
