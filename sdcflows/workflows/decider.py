import re
from future import raise_from

from nipype.pipelines import Workflow

from fmriprep.utils.misc import get_subject
from fmriprep.workflows.fieldmap.fieldmap_to_phasediff import fieldmap_to_phasediff
from fmriprep.workflows.fieldmap.se_pair_workflow import se_pair_workflow

class FieldmapDecider(Workflow):

    # POSSIBLE FILES ACCORDING TO BIDS 1.0.0
    # 8.9.1 one phase diff image, at least one magnitude image
    # 8.9.2 two phase images, two magnitude images
    # 8.9.3 fieldmap image (and one magnitude image)
    # 8.9.4 multiple phase encoded directions (topup)

    # search patterns
    suffixes = {
        'phasediff': r"phasediff\.nii",
        'magnitude': r"magnitude[0-9]*\.nii",
        'phase': r"phase[0-9]+\.nii",
        'fieldmap': r"fieldmap\.nii",
        'topup': r"epi\.nii"
    }

    def __init__(settings):
        subj_data = get_subject(settings['bids_root'], 
                                settings['subject_id'])

        try:
            a_filename = subj_data[next(iter(dictionary))]['fieldmaps'][0]
        except IndexError as e:
            raise_from("No fieldmap data found", e)

        for filename in subj_data[next(iter(subj_data))]['fieldmaps']:
            if re.search(suffixes['phasediff'], filename): # 8.9.1
                raise NotImplementedError("No workflow for phasediff fieldmap data")
            elif re.search(suffixes['phase']phase, filename): # 8.9.2
                raise NotImplementedError("No workflow for phase fieldmap data")
            elif re.search(suffixes['fieldmap'], a_filename): # 8.9.3
                return fieldmap_to_phasediff() # ???
            elif re.search(suffixes['topup'], a_filename): #8.0.4
                return se_pair_workflow() # ???

        raise IOError("Unrecognized fieldmap structure")
