#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
"""sdcflows setup script."""


def main():
    """Install entry-point."""
    from setuptools import setup

    setup(
        name="sdcflows", use_scm_version=True,
    )


if __name__ == "__main__":
    main()
