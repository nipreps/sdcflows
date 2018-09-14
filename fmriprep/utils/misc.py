#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utilities
"""


def remove_rotation_and_shear(img):
    from transforms3d.affines import decompose, compose
    import numpy as np

    T, _, Z, _ = decompose(img.affine)
    affine = compose(T=T, R=np.diag([1, 1, 1]), Z=Z)
    return img.__class__(np.asanyarray(img.dataobj), affine, img.header)


def split_and_rm_rotshear_func(in_file):
    import os
    from nilearn.image import iter_img
    out_files = []
    for i, img in enumerate(iter_img(in_file)):
        out_file = os.path.abspath('vol%04d.nii.gz' % i)
        img = remove_rotation_and_shear(img)
        img.to_filename(out_file)
        out_files.append(out_file)
    return out_files


def afni2itk_func(in_file):
    import os
    from numpy import loadtxt, hstack, vstack, zeros, float64

    def read_afni_affine(input_file, debug=False):
        orig_afni_mat = loadtxt(input_file)
        if debug:
            print(orig_afni_mat)

        output = []
        for i in range(orig_afni_mat.shape[0]):
            output.append(vstack((orig_afni_mat[i, :].reshape(3, 4, order='C'), [0, 0, 0, 1])))
        return output

    def get_ants_dict(affine, debug=False):
        out_dict = {}
        ants_affine_2d = hstack((affine[:3, :3].reshape(1, -1), affine[:3, 3].reshape(1, -1)))
        out_dict['AffineTransform_double_3_3'] = ants_affine_2d.reshape(-1, 1).astype(float64)
        out_dict['fixed'] = zeros((3, 1))
        if debug:
            print(out_dict)

        return out_dict

    out_file = os.path.abspath('mc4d.txt')
    with open(out_file, 'w') as fp:
        fp.write("#Insight Transform File V1.0\n")
        for i, affine in enumerate(read_afni_affine(in_file)):
            fp.write("#Transform %d\n" % i)
            fp.write("Transform: AffineTransform_double_3_3\n")
            trans_dict = get_ants_dict(affine)

            params_list = ["%g" % i for i in list(trans_dict['AffineTransform_double_3_3'])]
            params_str = ' '.join(params_list)
            fp.write("Parameters: " + params_str + "\n")

            fixed_params_list = ["%g" % i for i in list(trans_dict['fixed'])]
            fixed_params_str = ' '.join(fixed_params_list)
            fp.write("FixedParameters: " + ' '.join(fixed_params_str) + "\n")

    return out_file


def fix_multi_T1w_source_name(in_files):
    """
    Make up a generic source name when there are multiple T1s

    >>> fix_multi_T1w_source_name([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'])
    '/path/to/sub-045_T1w.nii.gz'

    """
    import os
    from nipype.utils.filemanip import filename_to_list
    base, in_file = os.path.split(filename_to_list(in_files)[0])
    subject_label = in_file.split("_", 1)[0].split("-")[1]
    return os.path.join(base, "sub-%s_T1w.nii.gz" % subject_label)


def add_suffix(in_files, suffix):
    """
    Wrap nipype's fname_presuffix to conveniently just add a prefix

    >>> add_suffix([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'], '_test')
    'sub-045_ses-test_T1w_test.nii.gz'

    """
    import os.path as op
    from nipype.utils.filemanip import fname_presuffix, filename_to_list
    return op.basename(fname_presuffix(filename_to_list(in_files)[0],
                                       suffix=suffix))


if __name__ == '__main__':
    pass
