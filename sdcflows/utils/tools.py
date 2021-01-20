"""Image processing tools."""


def brain_masker(in_file, out_file=None, padding=5):
    """Use grayscale morphological operations to obtain a quick mask of EPI data."""
    from pathlib import Path
    import nibabel as nb
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import ball
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed

    if out_file is None:
        out_file = Path("brainmask.nii.gz").absolute()

    # Load data
    img = nb.load(in_file)
    data = np.pad(img.get_fdata(dtype="float32"), padding)
    hdr = img.header.copy()

    # Cleanup background and invert intensity
    data[data < np.percentile(data[data > 0], 15)] = 0
    data *= -1.0
    data -= data.min()
    data /= data.max()

    # Grayscale closing to enhance CSF layer surrounding the brain
    closed = ndimage.grey_closing(data, structure=ball(1))
    th = threshold_otsu(closed) * .9

    # Rough binary mask
    closedbin = np.zeros_like(closed)
    closedbin[closed < th] = 1
    closedbin = ndimage.binary_opening(closedbin, ball(3)).astype("uint8")

    label_im, nb_labels = ndimage.label(closedbin)
    sizes = ndimage.sum(closedbin, label_im, range(nb_labels + 1))
    mask = sizes == sizes.max()
    closedbin = mask[label_im]
    closedbin = ndimage.binary_closing(closedbin, ball(5)).astype("uint8")
    closedbin_dil = ndimage.binary_dilation(closedbin, ball(3))

    closed[closed > 0.0] -= closed[closed > 0.0].min()
    closed *= 1.0 / closed.max()
    closed[(closed > 1.0 - 0.1) & ~closedbin_dil] = -1.0 * (
        closed[(closed > 1.0 - 0.1) & ~closedbin_dil] - 1.0
    )

    dist = ndimage.morphology.distance_transform_edt(
        ~(~ndimage.binary_erosion(closedbin, ball(5))
          * closedbin_dil)
    )
    dist = (-1.0 * (dist / dist.max()) + 1.0)
    hdr.set_data_dtype("float32")
    img.__class__(dist, img.affine, hdr).to_filename("dist.nii.gz")

    hdr.set_data_dtype("uint8")
    img.__class__(closedbin, img.affine, hdr).to_filename("closedbin.nii.gz")

    # Prepare data for watershed
    wshed = np.round(255 * closed / closed.max()).astype("uint16")
    hdr.set_data_dtype("uint16")
    img.__class__(wshed, img.affine, hdr).to_filename("wshed.nii.gz")
    markers = np.ones_like(closed, dtype="int8") * -1
    # markers[1:-1, 1:-1, 1:-1] = 0
    markers[closedbin_dil] = 0
    closed_eroded = ndimage.binary_erosion(closedbin, structure=ball(5))
    markers[closed_eroded] = 1

    # Run watershed
    labels = watershed(wshed, markers)[
        padding:-padding, padding:-padding, padding:-padding
    ]
    hdr.set_data_dtype("int16")
    img.__class__(markers.astype("int16"), img.affine, hdr).to_filename("markers.nii.gz")
    hdr.set_data_dtype("uint8")
    img.__class__((labels == 1).astype("uint8"), img.affine, hdr).to_filename(out_file)
    return str(out_file)
