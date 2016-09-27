'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''
from nipype.interfaces import utility
from nipype.pipeline import engine as pe

def discover_wf(name="ConfoundDiscoverer"):
    ''' Given a motion-corrected fMRI ('inputnode.fmri_file'),
           calculates global regressor, dvars, and tcompcor.
    If given movement parameters from MCFLIRT ('inputnode.movpar_file'),
           it also calculates frame displacement
    If given a segmentation ('inputnode.t1_seg'), it also calculates segment regressors and aCompCor

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
