#!/bin/bash
set -e

pushd scannertools_infra
pip3 install -e .
popd

for subdir in scannertools*/ ; do
    if [ "$subdir" = "scannertools_infra/"  ]; then
        continue
    fi

    pushd $subdir
    if [ "$tag2" = "cpu" ]; then
        CUDA_OPT=
    else
        CUDA_OPT=--install-option="--build-cuda=/usr/local/cuda"
    fi
    pip3 install ${CUDA_OPT} -e .
    popd
done
