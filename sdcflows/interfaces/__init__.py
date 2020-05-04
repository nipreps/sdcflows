# Backports


def dseg_label(in_seg, label, newpath=None):
    """Extract a particular label from a discrete segmentation."""
    from pathlib import Path
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix

    newpath = Path(newpath or '.')

    nii = nb.load(in_seg)
    data = np.int16(nii.dataobj) == label

    out_file = fname_presuffix(in_seg, suffix='_mask',
                               newpath=str(newpath.absolute()))
    new = nii.__class__(data, nii.affine, nii.header)
    new.set_data_dtype(np.uint8)
    new.to_filename(out_file)
    return out_file
