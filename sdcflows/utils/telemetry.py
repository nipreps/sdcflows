# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
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
# STATEMENT OF CHANGES: This file is derived from sources licensed under the Apache-2.0 terms,
# and this file has been changed.
# The original file this work derives from is found at:
# https://github.com/nipreps/mriqc/blob/8ceadba8669cc2a86119a97b9311ab968f11c6eb/mriqc/utils/telemetry.py
"""Ping MIGAS for telemetry."""
import migas

from sdcflows import config

MIGAS_PACKAGE = "nipreps/sdcflows"


def setup_migas(init_ping: bool = True, exit_ping: bool = True) -> None:
    """
    Prepare the migas python client to communicate with a migas server.
    If ``init`` is ``True``, send an initial breadcrumb.
    """
    # generate session UUID from generated run UUID
    session_id = None
    if config.execution.run_uuid:
        session_id = config.execution.run_uuid.split('_', 1)[-1]

    migas.setup(session_id=session_id)
    if init_ping:
        # send initial status ping
        send_crumb(status='R', status_desc='workflow start')
    if exit_ping:
        from migas.error.nipype import node_execution_error

        migas.track_exit(
            MIGAS_PACKAGE,
            config.environment.version,
            {'NodeExecutionError': node_execution_error},
        )


def send_crumb(**kwargs) -> dict:
    """
    Communicate with the migas telemetry server. This requires `migas.setup()` to be called.
    """
    return migas.add_breadcrumb(MIGAS_PACKAGE, config.environment.version, **kwargs)
