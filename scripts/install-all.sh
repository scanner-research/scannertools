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

    # Install dependencies first before package
    python3 setup.py egg_info
    requires_path="${subdir%/}.egg-info/requires.txt"
    if [ -f "${requires_path}" ]; then
        pip3 install -r ${requires_path}
    fi

    pip3 install ${CUDA_OPT} -e.
    popd
done
