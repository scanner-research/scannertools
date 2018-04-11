#!/bin/bash
set -e
pip install travis-sphinx tensorflow pytest
pytest
sphinx-apidoc -f -o doc/source tixelbox
travis-sphinx build -s doc -n
