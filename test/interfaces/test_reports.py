import os
import shutil
import tempfile
import unittest

from nipype.pipeline import engine as pe
from niworkflows.data.getters import get_mni_template_ras

from fmriprep.interfaces.reports import RegistrationRPT, BETRPT
from test.utils.tempdir import in_temporary_directory

MNI_DIR = get_mni_template_ras()

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

class TestBETRPT(unittest.TestCase):
    ''' tests it using mni as in_file '''

    def test_generate_report(self):
        ''' test of BET's report under a bunch of diff options for what to output'''
        boo = (True, False)
        for outline in boo:
            for mask in boo:
                for skull in boo:
                    self._smoke(BETRPT(in_file=os.path.join(MNI_DIR, 'MNI152_T1_2mm.nii.gz'),
                                       generate_report=True, outline=outline, mask=mask,
                                       skull=skull))

    def test_cannot_generate_report(self):
        ''' Can't generate a report if there are no nifti outputs. '''
        with self.assertRaises(Warning):
            self._smoke(BETRPT(in_file=os.path.join(MNI_DIR, 'MNI152_T1_2mm.nii.gz'),
                               generate_report=True, outline=False, mask=False, no_output=True))

    def test_generate_report_from_4d(self):
        ''' if the in_file was 4d, it should be able to produce the same report
        anyway (using arbitrary volume) '''
        pass

    @in_temporary_directory
    def _smoke(self, bet_interface):
        bet_interface.run()

        html_report = bet_interface.aggregate_outputs().html_report
        self.assertTrue(os.path.isfile(html_report), 'HTML report exists at {}'
                        .format(html_report))
