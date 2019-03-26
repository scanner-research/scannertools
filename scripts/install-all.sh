#!/bin/bash
set -e

pushd scannertools_infra
pip3 install -e .
popd

for subdir in scannertools*/ ; do
    echo "Test: $subdir"
    pushd $subdir
    if [ "$tag2" = "cpu" ]; then
        CUDA_OPT=
    else
        CUDA_OPT=--install-option="--build-cuda=/usr/local/cuda"
    fi
    pip3 install ${CUDA_OPT} -e .
    popd
done
