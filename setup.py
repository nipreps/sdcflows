#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
""" sdcflows setup script """


def main():
    """ Install entry-point """
    from setuptools import setup
    from sdcflows.__about__ import __version__, DOWNLOAD_URL

    import versioneer
    cmdclass = versioneer.get_cmdclass()

    setup(
        version=__version__,
        cmdclass=cmdclass,
        download_url=DOWNLOAD_URL,
        # Dependencies handling
        zip_safe=False,
    )


if __name__ == '__main__':
    main()
