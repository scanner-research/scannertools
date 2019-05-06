#!/bin/bash
set -e

for subdir in scannertools*/ ; do
    pushd $subdir
    rm -rf build *.egg-info ${subdir}build
    popd
done
