"""Test phase-difference type of fieldmaps."""
import pytest
from niworkflows.interfaces.bids import DerivativesDataSink
from nipype.pipeline import engine as pe

from ..phdiff import init_phdiff_wf, Workflow


@pytest.mark.parametrize('dataset', [
    'ds001600',
    'testdata',
])
def test_workflow(bids_layouts, tmpdir, output_path, dataset):
    """Test creation of the workflow."""
    tmpdir.chdir()

    data = bids_layouts[dataset]
    wf = Workflow(name='tstworkflow')
    phdiff_wf = init_phdiff_wf(omp_nthreads=1)
    phdiff_wf.inputs.inputnode.magnitude = data.get(
        suffix=['magnitude1', 'magnitude2'],
        acq='v4',
        return_type='file',
        extension=['.nii', '.nii.gz'])

    phdiff_file = data.get(suffix='phasediff', acq='v4',
                           extension=['.nii', '.nii.gz'])[0]

    phdiff_wf.inputs.inputnode.phasediff = phdiff_file.path
    phdiff_wf.inputs.inputnode.metadata = phdiff_file.get_metadata()

    if output_path:
        from ...interfaces.reportlets import FieldmapReportlet
        rep = pe.Node(FieldmapReportlet(), 'simple_report')

        dsink = pe.Node(DerivativesDataSink(
            base_directory=str(output_path), keep_dtype=True), name='dsink')
        dsink.interface.out_path_base = 'sdcflows'
        dsink.inputs.source_file = phdiff_file.path

        wf.connect([
            (phdiff_wf, rep, [
                ('outputnode.fmap', 'fieldmap'),
                ('outputnode.fmap_ref', 'reference'),
                ('outputnode.fmap_mask', 'mask')]),
            (rep, dsink, [('out_report', 'in_file')]),
        ])
    else:
        wf.add_nodes([phdiff_wf])

    wf.run()
