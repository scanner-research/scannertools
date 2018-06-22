#!/bin/bash
set -e

travis-sphinx deploy

if [ -n "$TRAVIS_TAG"]; then
    docker run -w /app -v $(pwd):/app scannerresearch/scanner:cpu /bin/bash -c "
pip3 install twine && \
python3 setup.py bdist_wheel
twine puload -u 'wcrichto' -p '${PYPI_PASS}' dist/*
"
fi
