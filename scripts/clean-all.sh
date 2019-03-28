#!/bin/bash
set -e

for subdir in scannertools*/ ; do
    pushd $subdir
    rm -rf build
    popd
done
