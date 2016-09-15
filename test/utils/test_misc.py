import os
import re
import unittest
from io import BytesIO
from zipfile import ZipFile

import pkg_resources as pkgr
from future.standard_library import install_aliases

import fmriprep.utils.misc as misc

install_aliases()
from urllib.request import urlopen



class TestCollectBids(unittest.TestCase):
    fake_data_root_location = os.path.join("test", "fake_data")
    fake_ds_location = os.path.join(fake_data_root_location,
                                    "BIDS-examples-1-enh-ds054",
                                    "ds054")
    subject_id = "sub-100185"

    @classmethod
    def setUp(cls):
        if not os.path.exists(cls.fake_data_root_location):
            print("Downloading test data")
            url = "http://github.com/chrisfilo/BIDS-examples-1/archive/enh/ds054.zip"
            with urlopen(url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall(cls.fake_data_root_location)

        cls.imaging_data = {
            cls.subject_id: misc.collect_bids_data(os.path.join(cls.fake_ds_location),
                                                   cls.subject_id)
        }

    def test_collect_bids_data(self):
        ''' test data has at least one subject with at least one session '''
        self.assertNotEqual(0, len(self.imaging_data))
        self.assertNotEqual(0, len(next(iter(self.imaging_data.values()))))

    def test_epi(self):
        epi_template = self.fake_ds_location + "/{subject}/func/{subject}_task-machinegame_run-06_bold.nii.gz"
        self.assert_key_exists(epi_template, 'func')

    def test_sbref(self):
        sbref_template = (self.fake_ds_location + "/{subject}/func/"
                          "{subject}_task-machinegame_run-06_sbref.nii.gz")
        self.assert_key_exists(sbref_template, 'sbref')

    def test_t1w(self):
        t1_template = self.fake_ds_location + "/{subject}/anat/{subject}_T1w.nii.gz"
        self.assert_key_exists(t1_template, 't1w')

    def test_fieldmaps(self):
        fieldmap_pattern = r"{0}\/fmap\/{0}_dir-[0-9]+_run-[0-9]+_epi\.nii\.gz"
        self.assert_fieldmap_files_exist(fieldmap_pattern, 'fieldmaps')
    
    # HELPER ASSERTIONS

    def assert_fieldmap_files_exist(self, pattern, key):
        for subject in self.imaging_data:
            search_pattern = pattern.format(subject)
            for fieldmap in self.imaging_data[subject][key]:
                match = re.search(search_pattern, fieldmap)
                self.assertTrue(match)

    def assert_key_exists(self, template, key):
        for subject in self.imaging_data:
            self.assertIn(template.format(subject=subject),
                          '\n'.join(self.imaging_data[subject][key]))

