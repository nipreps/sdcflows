import os
import numpy as np
import nibabel as nb
import unittest
from nilearn import image

from nipype.pipeline import engine as pe
from nipype.interfaces import IdentityInterface
from nipype.utils.tmpdirs import InTemporaryDirectory

from niworkflows.data.getters import get_mni_template_ras
from niworkflows.common.report_interfaces import (BETRPT, FLIRTRPT,
    RegistrationRPT, ApplyXFMRPT)

MNI_DIR = get_mni_template_ras()

@unittest.skip
class TestFLIRTRPT(unittest.TestCase):
    def setUp(self):
        self.out_file = "test_flirt.nii.gz"

    def test_known_file_out(self, flirt=FLIRTRPT):
        with InTemporaryDirectory():
            template = os.path.join(MNI_DIR, 'MNI152_T1_1mm.nii.gz')
            flirt_rpt = flirt(generate_report=True, in_file=template,
                              reference=template, out_file=self.out_file)
            flirt_rpt.run()
            html_report = flirt_rpt.aggregate_outputs().html_report
            self.assertTrue(os.path.isfile(html_report), 'HTML report exists at {}'
                            .format(html_report))

    def test_applyxfm_wrapper(self):
        self.test_known_file_out(ApplyXFMRPT)


class TestBETRPT(unittest.TestCase):
    ''' tests it using mni as in_file '''

    in_file = os.path.join(MNI_DIR, 'MNI152_T1_2mm.nii.gz')

    @unittest.skip
    def test_generate_report(self):
        ''' test of BET's report under basic (output binary mask) conditions '''
        self._smoke(BETRPT(in_file=self.in_file,
                           generate_report=True, mask=True))

    @unittest.skip
    def test_cannot_generate_report(self):
        ''' Can't generate a report if there are no nifti outputs. '''
        with self.assertRaises(Warning):
            self._smoke(BETRPT(in_file=self.in_file,
                               generate_report=True, outline=False, mask=False, no_output=True))

    @unittest.skip
    def test_generate_report_from_4d(self):
        ''' if the in_file was 4d, it should be able to produce the same report
        anyway (using arbitrary volume) '''
        # makeshift 4d in_file
        mni_file = self.in_file
        mni_4d = image.concat_imgs([mni_file, mni_file, mni_file])
        mni_4d_file = os.path.join(os.getcwd(), 'mni_4d.nii.gz')
        nb.save(mni_4d, mni_4d_file)

        self._smoke(BETRPT(in_file=mni_4d_file, generate_report=True, mask=True))

    def test_in_workflow(self):
        ''' test it in the workflow '''
        with InTemporaryDirectory():
            workflow = pe.Workflow(name="TestWorkflow")
            workflow.base_dir = os.getcwd()

            print("workdir = ", workflow.base_dir)

            inputnode = pe.Node(IdentityInterface(fields=['in_file']), name="inputnode")
            inputnode.inputs.in_file = self.in_file
            outputnode = pe.Node(IdentityInterface(fields=['html_report']), name="outputnode")

            betnode = pe.Node(BETRPT(generate_report=True, mask=True), name="betnode")

            workflow.connect([(inputnode, betnode, [('in_file', 'in_file')]),
                              (betnode, outputnode, [('html_report', 'html_report')])])

            workflow.run()

            #  html_report = betnode.interface.aggregate_outputs().html_report
            #  self.assertTrue(os.path.isfile(html_report), 'HTML report exists at {}'.format(html_report))

    def _smoke(self, bet_interface):
        with InTemporaryDirectory():
            bet_interface.run()

            html_report = bet_interface.aggregate_outputs().html_report
            self.assertTrue(os.path.isfile(html_report), 'HTML report exists at {}'
                            .format(html_report))
