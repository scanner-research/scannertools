#!/bin/bash
set -e

# Download non-pip dependencies
pushd /tmp
git clone https://github.com/davidsandberg/facenet
git clone https://github.com/scanner-research/rude-carnie
export PYTHONPATH=/tmp/facenet/src:/tmp/rude-carnie:$PYTHONPATH
popd

# Run unit tests
python3 setup.py test

# Build docs
pip3 install travis-sphinx sphinx-nameko-theme
sphinx-apidoc -f -o doc/source scannertools
travis-sphinx build -s doc -n
