# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:32:13 2016

@author: craigmoodie
"""

def anatomical_overlay(in_file, overlay_file, out_file):
    import os.path
    import matplotlib as mpl
    mpl.use('Agg')
    from nilearn.plotting import plot_anat as plot_anat
    mask_display = plot_anat(in_file)
    mask_display.add_edges(overlay_file)
    mask_display.dim = -1
    #  mask_display.add_contours(overlay_file)
    #  mask_display.add_overlay(overlay_file)
    mask_display.title(out_file, x=0.01, y=0.99, size=15, color=None,
                       bgcolor=None, alpha=1)
    mask_display.savefig(out_file)
    return os.path.abspath(out_file)


def parcel_overlay(in_file, overlay_file, out_file):
    import os.path
    import matplotlib as mpl
    mpl.use('Agg')
    from nilearn.plotting import plot_epi
    mask_display = plot_epi(in_file)
    mask_display.add_edges(overlay_file)
    #  mask_display.add_contours(overlay_file)
    #  mask_display.add_overlay(overlay_file)
    mask_display.title(out_file, x=0.01, y=0.99, size=15, color=None,
                       bgcolor=None, alpha=1)
    mask_display.savefig(out_file)
    return os.path.abspath(out_file)


def stripped_brain_overlay(in_file, overlay_file, out_file):
    import os.path
    import nibabel as nb
    import matplotlib as mpl
    mpl.use('Agg')
    from nilearn.plotting import plot_roi
    vmax = nb.load(in_file).get_data().reshape(-1).max()
    mask_display = plot_roi(
        in_file, overlay_file, output_file=out_file, title=out_file,
        display_mode="ortho", dim=-1, alpha=.3, vmax=vmax + 1)
    #  mask_display.bg_img(overlay_file)
    #  mask_display.title(out_file, x=0.01, y=0.99, size=15, color=None,
    #                     bgcolor=None, alpha=1)
    #  mask_display.display_mode = "yx"
    mask_display
    return os.path.abspath(out_file)
