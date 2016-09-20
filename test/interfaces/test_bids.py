import os
import shutil
import tempfile
import unittest

from nipype.pipeline import engine as pe

from fmriprep.interfaces.bids import DerivativesDataSink
import fmriprep.utils.misc as misc


class TestDerivativesDatasink(unittest.TestCase):
    prefix = 'sub-01_ses-01_'
    def setUp(self):
        _, self.infile_path = tempfile.mkstemp(prefix=self.prefix)
        fp = open(self.infile_path, 'w')
        fp.write('test text\n')
        fp.close()
        self.out_dir = tempfile.mkdtemp()
        print(self.infile_path)
        print(self.out_dir)

    def tearDown(self):
        os.remove(self.infile_path)
        #  shutil.rmtree(self.out_dir)
        
    def test_known_file_out(self):
        deriv_ds = pe.Node(
            DerivativesDataSink(
                base_directory=self.out_dir,
                source_file=self.infile_path,
                suffix='sfx',
                in_file=self.infile_path
            ),
            name='TestDerivatives'
        )
        deriv_ds.run()
        deriv_filename = self.infile_path.join('sfx')
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, 'deriv_filename)))
