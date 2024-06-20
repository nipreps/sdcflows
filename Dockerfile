# NiWorkflows Docker Container Image distribution
#
# MIT License
#
# Copyright (c) 2021 The NiPreps Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Ubuntu 22.04 LTS - Jammy
ARG BASE_IMAGE=ubuntu:jammy-20230308

#
# SDCFlows wheel
#
FROM python:slim AS src
RUN pip install build
RUN apt-get update && \
    apt-get install -y --no-install-recommends git
COPY . /src
RUN python -m build /src

#
# Download stages
#

# Utilities for downloading packages
FROM --platform=linux/amd64 ${BASE_IMAGE} as downloader

# Bump the date to current to refresh curl/certificates/etc
RUN echo "2023.11.09"
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    binutils \
                    bzip2 \
                    ca-certificates \
                    curl \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# AFNI
FROM downloader as afni
# Bump the date to current to update AFNI
RUN echo "2023.07.20"
RUN mkdir -p /opt/afni-latest \
    && curl -fsSL --retry 5 https://afni.nimh.nih.gov/pub/dist/tgz/linux_openmp_64.tgz \
    | tar -xz -C /opt/afni-latest --strip-components 1 \
    --exclude "linux_openmp_64/*.gz" \
    --exclude "linux_openmp_64/funstuff" \
    --exclude "linux_openmp_64/shiny" \
    --exclude "linux_openmp_64/afnipy" \
    --exclude "linux_openmp_64/lib/RetroTS" \
    --exclude "linux_openmp_64/lib_RetroTS" \
    --exclude "linux_openmp_64/meica.libs" \
    # Keep only what we use
    && find /opt/afni-latest -type f -not \( \
        -name "3dTshift" -or \
        -name "3dUnifize" -or \
        -name "3dAutomask" -or \
        -name "3dvolreg" \) -delete

# Convert3d 1.4.0
FROM downloader as c3d
RUN mkdir /opt/convert3d && \
    curl -fsSL --retry 5 https://sourceforge.net/projects/c3d/files/c3d/Experimental/c3d-1.4.0-Linux-gcc64.tar.gz/download \
    | tar -xz -C /opt/convert3d --strip-components 1

# Micromamba
FROM downloader as micromamba
WORKDIR /
# Bump the date to current to force update micromamba
RUN echo "2023.11.09"
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

ENV MAMBA_ROOT_PREFIX="/opt/conda"
COPY env.yml /tmp/env.yml
RUN micromamba create -y -f /tmp/env.yml && \
    micromamba clean -y -a

ENV PATH="/opt/conda/envs/sdcflows/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN /opt/conda/envs/sdcflows/bin/pip install --no-cache-dir -r /tmp/requirements.txt

#
# Main stage
#
FROM --platform=linux/amd64 ${BASE_IMAGE} as sdcflows

# Configure apt
ENV DEBIAN_FRONTEND="noninteractive" \
    LANG="en_US.UTF-8" \
    LC_ALL="en_US.UTF-8"

# Some baseline tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    ca-certificates \
                    curl \
                    git \
                    gnupg \
                    lsb-release \
                    netbase \
                    xvfb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure PPAs for libpng12 and libxp6
RUN GNUPGHOME=/tmp gpg --keyserver hkps://keyserver.ubuntu.com --no-default-keyring --keyring /usr/share/keyrings/linuxuprising.gpg --recv 0xEA8CACC073C3DB2A \
    && GNUPGHOME=/tmp gpg --keyserver hkps://keyserver.ubuntu.com --no-default-keyring --keyring /usr/share/keyrings/zeehio.gpg --recv 0xA1301338A3A48C4A \
    && echo "deb [signed-by=/usr/share/keyrings/linuxuprising.gpg] https://ppa.launchpadcontent.net/linuxuprising/libpng12/ubuntu jammy main" > /etc/apt/sources.list.d/linuxuprising.list \
    && echo "deb [signed-by=/usr/share/keyrings/zeehio.gpg] https://ppa.launchpadcontent.net/zeehio/libxp/ubuntu jammy main" > /etc/apt/sources.list.d/zeehio.list

# Dependencies for AFNI; requires a discontinued multiarch-support package from bionic (18.04)
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
           ed \
           gsl-bin \
           libglib2.0-0 \
           libglu1-mesa-dev \
           libglw1-mesa \
           libgomp1 \
           libjpeg62 \
           libpng12-0 \
           libxm4 \
           libxp6 \
           netpbm \
           tcsh \
           xfonts-base \
           xvfb \
    && curl -sSL --retry 5 -o /tmp/multiarch.deb http://archive.ubuntu.com/ubuntu/pool/main/g/glibc/multiarch-support_2.27-3ubuntu1.5_amd64.deb \
    && dpkg -i /tmp/multiarch.deb \
    && rm /tmp/multiarch.deb \
    && apt-get install -f \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && gsl2_path="$(find / -name 'libgsl.so.19' || printf '')" \
    && if [ -n "$gsl2_path" ]; then \
         ln -sfv "$gsl2_path" "$(dirname $gsl2_path)/libgsl.so.0"; \
    fi \
    && ldconfig

# Install files from stages
COPY --from=afni /opt/afni-latest /opt/afni-latest
COPY --from=c3d /opt/convert3d/bin/c3d_affine_tool /usr/bin/c3d_affine_tool

# AFNI config
ENV PATH="/opt/afni-latest:$PATH" \
    AFNI_IMSAVE_WARNINGS="NO" \
    AFNI_PLUGINPATH="/opt/afni-latest"

# Create a shared $HOME directory
RUN useradd -m -s /bin/bash -G users sdcflows
WORKDIR /home/sdcflows
ENV HOME="/home/sdcflows" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
COPY .docker/files/nipype.cfg $HOME/.nipype/nipype.cfg

COPY --from=micromamba /bin/micromamba /bin/micromamba
COPY --from=micromamba /opt/conda/envs/sdcflows /opt/conda/envs/sdcflows

ENV MAMBA_ROOT_PREFIX="/opt/conda"
RUN micromamba shell init -s bash && \
    echo "micromamba activate sdcflows" >> $HOME/.bashrc
ENV PATH="/opt/conda/envs/sdcflows/bin:$PATH" \
    CPATH="/opt/conda/envs/sdcflows/include:$CPATH" \
    LD_LIBRARY_PATH="/opt/conda/envs/sdcflows/lib:$LD_LIBRARY_PATH"

# FSL environment
ENV LANG="C.UTF-8" \
    LC_ALL="C.UTF-8" \
    PYTHONNOUSERSITE=1 \
    FSLDIR="/opt/conda/envs/sdcflows" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    FSLLOCKDIR="" \
    FSLMACHINELIST="" \
    FSLREMOTECALL="" \
    FSLGECUDAQ="cuda.q"

# Open a hook to patch FreeSurfer in
ENV FREESURFER_HOME="/opt/freesurfer" \
    MINC_LIB_DIR="/opt/freesurfer/mni/lib"
ENV SUBJECTS_DIR="$FREESURFER_HOME/subjects" \
    FUNCTIONALS_DIR="$FREESURFER_HOME/sessions" \
    MNI_DIR="$FREESURFER_HOME/mni" \
    LOCAL_DIR="$FREESURFER_HOME/local" \
    MINC_BIN_DIR="$FREESURFER_HOME/mni/bin" \
    MNI_DATAPATH="$FREESURFER_HOME/mni/data" \
    PATH="$FREESURFER_HOME/bin:$MINC_BIN_DIR:$PATH"

# Unless otherwise specified each process should only use one thread - nipype
# will handle parallelization
ENV MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

# Installing SDCFlows
COPY --from=src /src/dist/*.whl .
RUN pip install --no-cache-dir $( ls *.whl )[all]

RUN find $HOME -type d -exec chmod go=u {} + && \
    find $HOME -type f -exec chmod go=u {} +

ENV IS_DOCKER_8395080871=1

RUN ldconfig
WORKDIR /tmp/

ARG VERSION
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="SDCflows" \
      org.label-schema.description="SDCflows - Susceptibility Distortion Correction (SDC) workflows for EPI MR schemes" \
      org.label-schema.url="http://sdcflows.readthedocs.io" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/nipreps/sdcflows" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"
