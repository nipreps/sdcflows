# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
"""Visualization tooling."""


def plot_registration(
    anat_nii,
    div_id,
    plot_params=None,
    order=('z', 'x', 'y'),
    cuts=None,
    estimate_brightness=False,
    label=None,
    contour=None,
    compress='auto',
    overlay=None,
    overlay_params=None,
):
    """
    Plot the foreground and background views.

    Default order is: axial, coronal, sagittal
    """
    from uuid import uuid4

    import matplotlib.pyplot as plt
    from lxml import etree
    from nilearn.plotting import plot_anat
    from nireports._vendored.svgutils.transform import SVGFigure
    from nireports.reportlets.utils import SVGNS, extract_svg, robust_set_limits

    plot_params = plot_params or {}

    # Use default MNI cuts if none defined
    if cuts is None:
        raise NotImplementedError  # TODO

    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(
            anat_nii.get_fdata(dtype='float32').reshape(-1), plot_params
        )

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
        if overlay is not None:
            _overlay_params = {
                'vmin': overlay.get_fdata(dtype='float32').min(),
                'vmax': overlay.get_fdata(dtype='float32').max(),
                'cmap': plt.cm.gray,
                'interpolation': 'nearest',
            }
            _overlay_params.update(overlay_params)
            display.add_overlay(overlay, **_overlay_params)
        if contour is not None:
            display.add_contours(contour, colors='g', levels=[0.5], linewidths=0.5)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        xml_data = etree.fromstring(svg)
        find_text = etree.ETXPath("//{%s}g[@id='figure_1']" % SVGNS)
        find_text(xml_data)[0].set('id', '%s-%s-%s' % (div_id, mode, uuid4()))

        svg_fig = SVGFigure()
        svg_fig.root = xml_data
        out_files.append(svg_fig)

    return out_files


def coolwarm_transparent(max_alpha=0.7, opaque_perc=30, transparent_perc=8):
    """Modify the coolwarm color scale to have full transparency around the middle."""
    import matplotlib.pylab as pl
    import numpy as np
    from matplotlib.colors import ListedColormap

    # Choose colormap
    cmap = pl.cm.coolwarm

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    _20perc = (cmap.N * opaque_perc) // 100
    midpoint = cmap.N // 2 + 1
    _10perc = (cmap.N * transparent_perc) // 100
    # Set alpha
    alpha = np.zeros(cmap.N)
    alpha[:_20perc] = max_alpha
    alpha[-_20perc:] = max_alpha
    alpha[_20perc : midpoint - _10perc - 1] = np.linspace(
        max_alpha, 0, len(alpha[_20perc : midpoint - _10perc - 1])
    )
    alpha[midpoint + _10perc : -_20perc] = np.linspace(
        0, max_alpha, len(alpha[midpoint + _10perc : -_20perc])
    )
    my_cmap[:, -1] = alpha
    return ListedColormap(my_cmap)
