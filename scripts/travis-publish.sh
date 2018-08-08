#!/bin/bash
set -e

pip3 install travis-sphinx

docker run -v $(pwd):/opt/scannertools $DOCKER_REPO:cpu-latest bash -c "
cd /opt/scannertools

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
pip3 install travis-sphinx sphinx-nameko-theme

sphinx-apidoc -f -o doc/source scannertools
travis-sphinx build -s doc -n
"

travis-sphinx deploy

if [ -n "$TRAVIS_TAG" ];
then
    docker run -v $(pwd):/opt/scannertools $DOCKER_REPO:cpu-latest bash -c "
cd /opt/scannertools
pip3 install twine
python3 setup.py bdist_wheel
twine upload -u 'wcrichto' -p '${PYPI_PASS}' dist/*
"
fi
