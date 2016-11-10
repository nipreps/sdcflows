import os
import shutil
import tempfile
import unittest

from nipype.pipeline import engine as pe

from fmriprep.interfaces.reports import RegistrationRPT


class TestFLIRTRPT(unittest.TestCase):
    prefix = 'sub-01_ses-01_'
    def setUp(self):
        _, self.infile_path = tempfile.mkstemp(prefix=self.prefix)
        self.infile_contents = 'test text\n'
        fp = open(self.infile_path, 'w')
        fp.write(self.infile_contents)
        fp.close()
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        os.remove(self.infile_path)
        shutil.rmtree(self.out_dir)
        
    def test_known_file_out(self):
        suffix = 'sfx'
        flirt_rpt = pe.Node(
            FLIRTRPT(
            ),
            name='TestFLIRTRPT'
        )
        flirt_rpt.run()
        
        flirt_rpt_path = os.path.join(self.out_dir, out_file)
        self.assertTrue(os.path.isfile(flirt_rpt_path))
