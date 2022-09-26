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
"""Basic miscellaneous utilities."""
import logging


def front(inlist):
    """
    Pop from a list or tuple, otherwise return untouched.

    Examples
    --------
    >>> front([1, 0])
    1

    >>> front("/path/somewhere")
    '/path/somewhere'

    """
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def last(inlist):
    """
    Return the last element from a list or tuple, otherwise return untouched.

    Examples
    --------
    >>> last([1, 0])
    0

    >>> last("/path/somewhere")
    '/path/somewhere'

    """
    if isinstance(inlist, (list, tuple)):
        return inlist[-1]
    return inlist


def get_free_mem():
    """Probe the free memory right now."""
    try:
        from psutil import virtual_memory

        return round(virtual_memory().free, 1)
    except Exception:
        return None


def create_logger(name: str, level: int = 40) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # clear any existing handlers
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    # formatter = logging.Formatter('[%(name)s %(asctime)s] - %(levelname)s: %(message)s')
    formatter = logging.Formatter('[%(name)s - %(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
