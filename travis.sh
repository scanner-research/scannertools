#!/bin/bash

pip install travis-sphinx tensorflow
sphinx-apidoc -f -o doc/source tixelbox
travis-sphinx build -s doc -n
