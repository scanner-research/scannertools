#!/bin/bash
set -e

for subdir in scannertools*/ ; do
    if [ "$subdir" = "scannertools_infra"  ]; then
        continue
    fi

    pushd $subdir
    python3 setup.py test
    popd
done
