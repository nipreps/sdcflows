import json
import misc
import re
import unittest

class TestCollectBids(unittest.TestCase):
    def setUp(self):
        self.dataset = "../../test_data/aa_conn"
        self.imaging_data = misc.collect_bids_data(self.dataset)

    def test_epi(self):
        epi_template = "{subject}/func/{subject}_task-rest_acq-LR_run-1_bold.nii.gz"
        for subject in self.imaging_data:
            for session in self.imaging_data[subject]:
                self.assertIn(epi_template.format(subject=subject), 
                              self.imaging_data[subject][session]['epi'])

    def test_epi_meta(self):
        epi_meta_template = "{subject}/func/{subject}_task-rest_acq-LR_run-1_bold.json"
        for subject in self.imaging_data:
            for session in self.imaging_data[subject]:
                self.assertIn(epi_meta_template.format(subject=subject), 
                              self.imaging_data[subject][session]['epi_meta'])

    def test_sbref(self):
        sbref_template = "{subject}/func/{subject}_task-rest_acq-LR_run-1_sbref.nii.gz"
        for subject in self.imaging_data:
            for session in self.imaging_data[subject]:
                self.assertIn(sbref_template.format(subject=subject), 
                              self.imaging_data[subject][session]['sbref'])

    def test_sbref_meta(self):
        sbref_meta_template = "{subject}/func/{subject}_task-rest_acq-LR_run-1_sbref.json"
        for subject in self.imaging_data:
            for session in self.imaging_data[subject]:
                self.assertIn(sbref_meta_template.format(subject=subject), 
                              self.imaging_data[subject][session]['sbref_meta'])
    
    def test_t1(self):
        t1_template = "{subject}/anat/{subject}_run-1_T1w.nii.gz"
        for subject in self.imaging_data:
            for session in self.imaging_data[subject]:
                self.assertIn(t1_template.format(subject=subject), 
                              self.imaging_data[subject][session]['t1'])

    def test_fieldmaps(self):
        for subject in self.imaging_data:
            fieldmap_pattern = r"{0}\/fmap\/{0}_dir-[0-9]+_run-[0-9]+_epi\.nii\.gz".format(subject)
            for session in self.imaging_data[subject]:
                for fieldmap in self.imaging_data[subject][session]['fieldmaps']:
                    match = re.search(fieldmap_pattern, fieldmap)
                    self.assertTrue(match)
    
        
if __name__ == '__main__':
    unittest.main() 
    #dataset = "../../test_data/aa_conn"
    #imaging_data = misc.collect_bids_data(dataset)
