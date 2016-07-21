#!/bin/bash

set -x
set -e

docker run -t -v ~/scratch/data:/root/src/preprocessing-workflow/test_data --entrypoint="/usr/bin/run_unittests" -w "/root/src/preprocessing-workflow" oesteban/fmriprep

docker run -i -v /etc/localtime:/etc/localtime:ro -v ~/scratch:/scratch -w /scratch --entrypoint="/usr/bin/run_fmriprep" oesteban/fmriprep -B /scratch/data/AA_Conn -S S5271NYO -o /scratch/outputs/ms-fmriprep/out -w /scratch/outputs/ms-fmriprep/work --debug
