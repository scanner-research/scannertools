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
        INSTALL_TAG=cpu
    else
        CUDA_OPT=--install-option="--build-cuda=/usr/local/cuda"
        INSTALL_TAG=gpu
    fi
    pip3 install -v ${CUDA_OPT} -e ".[${INSTALL_TAG}]"
    popd
done
