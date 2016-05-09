#!/bin/bash

set -x
set -e

docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch --entrypoint="/usr/bin/run_fmriprep" oesteban/fmriprep -i /scratch/data/aa_conn -o outputs/ms-fmriprep/out -w outputs/ms-fmriprep/work