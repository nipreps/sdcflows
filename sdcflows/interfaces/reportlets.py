# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Interfaces to generate speciality reportlets."""
from nilearn.image import threshold_img, load_img
from niworkflows import NIWORKFLOWS_LOG
from niworkflows.viz.utils import cuts_from_bbox, compose_view
from nipype.interfaces.base import File, isdefined
from nipype.interfaces.mixins import reporting


class FieldmapReportletInputSpec(reporting.ReportCapableInputSpec):
    reference = File(exists=True, mandatory=True, desc='input reference')
    fieldmap = File(exists=True, mandatory=True, desc='input fieldmap')
    mask = File(exists=True, desc='brain mask')


class FieldmapReportlet(reporting.ReportCapableInterface):
    """An abstract mixin to registration nipype interfaces."""

    _n_cuts = 7
    input_spec = FieldmapReportletInputSpec
    output_spec = reporting.ReportCapableOutputSpec

    def __init__(self, **kwargs):
        """Instantiate FieldmapReportlet."""
        self._n_cuts = kwargs.pop('n_cuts', self._n_cuts)
        super(FieldmapReportlet, self).__init__(generate_report=True, **kwargs)

    def _run_interface(self, runtime):
        return runtime

    def _generate_report(self):
        """Generate a reportlet."""
        NIWORKFLOWS_LOG.info('Generating visual report')

        refnii = load_img(self.inputs.reference)
        fmapnii = load_img(self.inputs.fieldmap)
        contour_nii = load_img(self.inputs.mask) if isdefined(self.inputs.mask) else None
        mask_nii = threshold_img(refnii, 1e-3)
        cuts = cuts_from_bbox(contour_nii or mask_nii, cuts=self._n_cuts)

        # Call composer
        compose_view(
            plot_registration(refnii, 'fixed-image',
                              estimate_brightness=True,
                              cuts=cuts,
                              label='reference',
                              contour=contour_nii,
                              compress=False),
            plot_registration(fmapnii, 'moving-image',
                              estimate_brightness=True,
                              cuts=cuts,
                              label='fieldmap (Hz)',
                              contour=contour_nii,
                              compress=False,
                              plot_params={'cmap': 'coolwarm'}),
            out_file=self._out_report
        )

def plot_registration(anat_nii, div_id, plot_params=None,
                      order=('z', 'x', 'y'), cuts=None,
                      estimate_brightness=False, label=None, contour=None,
                      compress='auto'):
    """
    Plots the foreground and background views
    Default order is: axial, coronal, sagittal
    """
    from uuid import uuid4

    from lxml import etree
    from nilearn.plotting import plot_anat
    from svgutils.transform import SVGFigure
    from niworkflows.viz.utils import robust_set_limits, extract_svg, SVGNS

    plot_params = plot_params or {}

    # Use default MNI cuts if none defined
    if cuts is None:
        raise NotImplementedError  # TODO

    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(anat_nii.get_data().reshape(-1),
                                        plot_params)

    # Plot each cut axis
    for i, mode in enumerate(list(order)):
        plot_params['display_mode'] = mode
        plot_params['cut_coords'] = cuts[mode]
        if i == 0:
            plot_params['title'] = label
        else:
            plot_params['title'] = None

        # Generate nilearn figure
        display = plot_anat(anat_nii, **plot_params)
        if contour is not None:
            display.add_contours(contour, colors='g', levels=[0.5],
                                 linewidths=0.5)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        try:
            xml_data = etree.fromstring(svg)
        except etree.XMLSyntaxError as e:
            NIWORKFLOWS_LOG.info(e)
            return
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % SVGNS)
        find_text(xml_data)[0].set('id', '%s-%s-%s' % (div_id, mode, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files
