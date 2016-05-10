#!/bin/bash

set -x
set -e

docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch --entrypoint="/usr/bin/run_fmriprep" oesteban/fmriprep -B /scratch/data/AA_Conn -S S8848XEU -o outputs/ms-fmriprep/out -w outputs/ms-fmriprep/work