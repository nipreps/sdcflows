import os
import shutil
import tempfile
import unittest

from nipype.interfaces import fsl
from nipype.pipeline import engine as pe
from niworkflows.data import get_mni_template

from fmriprep.interfaces.reports import FLIRTRPT
#  from fmriprep.test.utils.tempdir import execute_in_temporary_directory

class TestFLIRTRPT(unittest.TestCase):
    prefix = 'sub-01_ses-01_'
    def setUp(self):
        _, self.infile_path = tempfile.mkstemp(prefix=self.prefix)
        self.infile_contents = 'test text\n'
        fp = open(self.infile_path, 'w')
        fp.write(self.infile_contents)
        fp.close()
        self.out_dir = tempfile.mkdtemp()
        self.out_file = "test_flirt.nii.gz"

    def tearDown(self):
        os.remove(self.infile_path)
        shutil.rmtree(self.out_dir)
        
    def test_known_file_out(self):
        suffix = 'sfx'
        flirt_rpt = pe.Node(
            FLIRTRPT(),
            name='TestFLIRTRPT'
        )
        flirt_rpt.inputs.reference = os.path.join(get_mni_template(),
                                                   'MNI152_T1_1mm.nii.gz')
        flirt_rpt.inputs.in_file = os.path.join(get_mni_template(),
                                           'MNI152_T1_1mm.nii.gz')
        flirt_rpt.inputs.out_file = self.out_file
        flirt_rpt.run()
        flirt_rpt_path = os.path.join(self.out_dir, self.out_file)
        self.assertTrue(os.path.isfile(flirt_rpt_path))

#  @execute_in_temporary_directory
#class TestBETRPT(unittest.TestCase):
#    ''' tests it using epi as in_file '''
#
#    def test_generate_report(self):
#        ''' test of BET's report under a bunch of diff options for what to output'''
#        boo = (True, False)
#
#        for outline in boo:
#            for mask in boo:
#                for skull in boo:
#                    for no_output in boo:
#                        self._smoke(fsl.BET(in_file='epi', outline=outline, mask=mask, skull=skull,
#                                            no_output=no_output))
#
#    def _smoke(self, bet_interface):
#        bet_interface.run()
#
#        self.assert_true(os.path.isfile(bet_interface.outputs.html_report),
#                         'HTML report exists at {}'.format(bet_interface.outputs.html_report))
