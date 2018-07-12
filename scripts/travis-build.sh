#!/bin/bash
set -e

pushd /tmp
git clone https://github.com/davidsandberg/facenet
git clone https://github.com/scanner-research/rude-carnie
export PYTHONPATH=/tmp/facenet/src/align:/tmp/rude-carnie:$PYTHONPATH
popd

pip3 install travis-sphinx sphinx-nameko-theme pytest tensorflow==1.5.0
pytest tests -vv
sphinx-apidoc -f -o doc/source scannertools
travis-sphinx build -s doc -n
