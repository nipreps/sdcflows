'''
Workflow for discovering confounds.
Calculates frame displacement, segment regressors, global regressor, dvars, aCompCor, tCompCor
'''

def discover_wf(settings=None):
    ''' Given an fMRI, calculates global regressor, dvars, and tcompcor.
    If given movement parameters, it also calculates frame displacement
    If given a segmentation, it also calculates segment regressors and aCompCor '''
    return settings
