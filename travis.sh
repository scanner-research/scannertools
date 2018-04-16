#!/bin/bash
set -e
pip install travis-sphinx tensorflow==1.5.0 pytest
pytest tests -vv
sphinx-apidoc -f -o doc/source scannertools
travis-sphinx build -s doc -n
