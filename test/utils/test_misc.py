import fmriprep.utils.misc as misc
import re
import os
import unittest
import pkg_resources as pkgr
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile



class TestCollectBids(unittest.TestCase):
    fake_data_location = os.path.join("test", "fake_data")
    subject_id = "100185"

    @classmethod
    def setUp(cls):
        if not os.path.exists(cls.fake_data_location):
            print("Downloading test data")
            url = "http://github.com/chrisfilo/BIDS-examples-1/archive/enh/ds054.zip"
            with urlopen(url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall(cls.fake_data_location)

        cls.imaging_data = {
            cls.subject_id: misc.collect_bids_data(os.path.join(cls.fake_data_location,
                                                                "BIDS-examples-1-enh-ds054",
                                                               "ds054"),
                                                   cls.subject_id,
                                                   spec=pkgr.resource_filename(
                                                       'fmriprep',
                                                       'data/bids.json'))
        }
        print(cls.imaging_data)



    def test_collect_bids_data(self):
        ''' test data has at least one subject with at least one session '''
        self.assertNotEqual(0, len(self.imaging_data))
        self.assertNotEqual(0, len(next(iter(self.imaging_data.values()))))

    def test_epi(self):
        epi_template = "{subject}/func/{subject}_task-rest_acq-LR_run-1_bold.nii.gz"
        self.assert_key_exists(epi_template, 'epi')

    def test_sbref(self):
        sbref_template = (self.fake_data_location + "/{subject}/func/"
                          "{subject}_task-rest_acq-LR_run-1_sbref.nii.gz")
        self.assert_key_exists(sbref_template, 'sbref')

    def test_t1(self):
        t1_template = "{subject}/anat/{subject}_run-1_T1w.nii.gz"
        self.assert_key_exists(t1_template, 't1')

    def test_fieldmaps(self):
        fieldmap_pattern = r"{0}\/fmap\/{0}_dir-[0-9]+_run-[0-9]+_epi\.nii\.gz"
        self.assert_fieldmap_files_exist(fieldmap_pattern, 'fieldmaps')
    
    # HELPER ASSERTIONS

    def assert_fieldmap_files_exist(self, pattern, key):
        for subject in self.imaging_data:
            search_pattern = pattern.format(subject)
            for session in self.imaging_data[subject]:
                for fieldmap in self.imaging_data[subject][session][key]:
                    match = re.search(search_pattern, fieldmap)
                    self.assertTrue(match)

    def assert_key_exists(self, template, key):
        for subject in self.imaging_data:
            for session in self.imaging_data[subject]:
                self.assertIn(template.format(subject=subject),
                              self.imaging_data[subject][session][key])
        
if __name__ == '__main__':
    unittest.main() 
    #dataset = "../../test_data/aa_conn"
    #imaging_data = misc.collect_bids_data(dataset)
