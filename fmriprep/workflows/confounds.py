'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''
from nipype.interfaces import utility
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

    workflow = pe.Workflow(name=name)
    workflow.connect([
        (inputnode, outputnode, [('fmri_file', 'confounds_file_name')])
    ])

    return workflow
