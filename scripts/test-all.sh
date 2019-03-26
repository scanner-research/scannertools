#!/bin/bash
set -e

for subdir in scannertools*/ ; do
    pushd $subdir
    python3 setup.py test
    popd
done
