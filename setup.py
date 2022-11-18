#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
"""sdcflows setup script."""


def main():
    """Install entry-point."""
    from setuptools import setup, Extension
    from Cython.Build import cythonize
    from numpy import get_include

    extensions = [
        Extension(
            "sdcflows.lib.bspline",
            ["sdcflows/lib/bspline.pyx"])
    ]

    setup(
        name="sdcflows",
        use_scm_version=True,
        ext_modules=cythonize(extensions),
        include_dirs=[get_include()],
    )


if __name__ == "__main__":
    main()
