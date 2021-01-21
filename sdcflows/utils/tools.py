"""Image processing tools."""


def brain_masker(in_file, out_file=None, padding=5):
    """Use grayscale morphological operations to obtain a quick mask of EPI data."""
    from pathlib import Path
    import nibabel as nb
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import ball
    from skimage.filters import threshold_otsu
    from skimage.segmentation import random_walker

    # Load data
    img = nb.load(in_file)
    data = np.pad(img.get_fdata(dtype="float32"), padding)
    hdr = img.header.copy()

    # Cleanup background and invert intensity
    data[data < np.percentile(data[data > 0], 15)] = 0
    data[data > 0] -= data[data > 0].min()
    datainv = -data.copy()
    datainv -= datainv.min()
    datainv /= datainv.max()

    # Grayscale closing to enhance CSF layer surrounding the brain
    closed = ndimage.grey_closing(datainv, structure=ball(1))
    th = threshold_otsu(closed) * 0.9

    # Rough binary mask
    closedbin = np.zeros_like(closed)
    closedbin[closed < th] = 1
    closedbin = ndimage.binary_opening(closedbin, ball(3)).astype("uint8")

    label_im, nb_labels = ndimage.label(closedbin)
    sizes = ndimage.sum(closedbin, label_im, range(nb_labels + 1))
    mask = sizes == sizes.max()
    closedbin = mask[label_im]
    closedbin = ndimage.binary_closing(closedbin, ball(5)).astype("uint8")

    # Prepare markers
    markers = np.ones_like(closed, dtype="int8") * 2
    markers[1:-1, 1:-1, 1:-1] = 0
    closedbin_dil = ndimage.binary_dilation(closedbin, ball(5))
    markers[closedbin_dil] = 0
    closed_eroded = ndimage.binary_erosion(closedbin, structure=ball(5))
    markers[closed_eroded] = 1

    # Run random walker
    closed[closed > 0.0] -= closed[closed > 0.0].min()
    segtarget = (2 * closed / closed.max()) - 1.0
    labels = random_walker(
        segtarget, markers, spacing=img.header.get_zooms()[:3], return_full_prob=True
    )[..., padding:-padding, padding:-padding, padding:-padding]

    if out_file is None:
        out_probseg = Path("brain_probseg.nii.gz").absolute()
        out_mask = Path("brain_mask.nii.gz").absolute()

    hdr.set_data_dtype("float32")
    img.__class__((labels[0, ...]), img.affine, hdr).to_filename(out_probseg)

    hdr.set_data_dtype("uint8")
    img.__class__((labels[0, ...] >= 0.5).astype("uint8"), img.affine, hdr).to_filename(
        out_mask
    )
    return str(out_probseg), str(out_mask)
