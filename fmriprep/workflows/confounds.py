'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''
from nipype.interfaces import utility, nilearn
from nipype.pipeline import engine as pe

def discover_wf(name="ConfoundDiscoverer"):
    ''' All input fields are required.

    Calculates global regressor, dvars, and tcompcor
        from motion-corrected fMRI ('inputnode.fmri_file').
    Calculates frame displacement from MCFLIRT movement parameters ('inputnode.movpar_file')
    Calculates segment regressors and aCompCor
        from a white matter/gray matter/CSF segmentation ('inputnode.t1_seg')

    Saves the confounds in a file ('outputnode.confounds_file')'''

    inputnode = pe.Node(utility.IdentityInterface(fields=['fmri_file', 'movpar_file', 't1_seg']),
                        name='inputnode')
    outputnode = pe.Node(utility.IdentityInterface(fields=['confounds_file']),
                         name='outputnode')

    # Global and segment regressors
    signals = pe.Node(nilearn.SignalExtraction(include_global=True, detrend=True,
                                       class_labels=['white matter', 'gray matter', 'CSF']), # check
                      name="SignalExtraction")

    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, signals, [('fmri_file', 'in_file'),
                              ('t1_seg', 'label_files')]),
        (signals, outputnode, [('out_file', 'confounds_file_name')])
    ])

    return workflow
