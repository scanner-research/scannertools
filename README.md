# scannertools: video processing toolkit &nbsp; [![Build Status](https://travis-ci.org/scanner-research/scannertools.svg?branch=master)](https://travis-ci.org/scanner-research/scannertools)

Scannertools is a Python library of easy-to-use, off-the-shelf pipelines written using the [Scanner](https://github.com/scanner-research/scanner/) video processing engine. Scannertools provides implementations of:

* [Object detection](http://scanner.run/api/scannertools.html#object-detection)
* [Face detection](http://scanner.run/api/scannertools.html#face-detection)
* [Face embedding](http://scanner.run/api/scannertools.html#face-embedding)
* [Gender detection](http://scanner.run/api/scannertools.html#gender-detection)
* [Pose detection](http://scanner.run/api/scannertools.html#pose-detection)
* [Optical flow](http://scanner.run/api/scannertools.html#optical-flow)
* [Shot detection](http://scanner.run/api/scannertools.html#shot-detection)

See the documentation on [scanner.run](http://scanner.run/api.html#scannertools-the-scanner-standard-library) for more details.

## Installation

You must have Scanner and all of its dependencies installed. See our [installation guide](http://scanner.run/guide/getting-started.html).

Each subdirectory prefixed with "scannertools" is a module containing Python and C++ Scanner ops. The modules are separated because each expects different system dependencies:

* [scannertools](https://github.com/scanner-research/scannertools/tree/master/scannertools): no additional dependencies beyond Scanner.
  * [scannertools.face_detection](https://github.com/scanner-research/scannertools/blob/master/scannertools/scannertools/face_detection.py) and [scannertools.face_embedding](https://github.com/scanner-research/scannertools/blob/master/scannertools/scannertools/face_embedding.py): depends on [Facenet](https://github.com/davidsandberg/facenet)
  * [scannertools.gender_detection](https://github.com/scanner-research/scannertools/blob/master/scannertools/scannertools/gender_detection.py): depends on [rude-carnie](https://github.com/dpressel/rude-carnie)
  * [scannertools.object_detection](https://github.com/scanner-research/scannertools/blob/master/scannertools/scannertools/object_detection.py): depends on [TensorFlow](https://www.tensorflow.org/)
* [scannertools_caffe](https://github.com/scanner-research/scannertools/tree/master/scannertools_caffe): depends on [Caffe](http://caffe.berkeleyvision.org/installation.html).
* [scannertools_sql](https://github.com/scanner-research/scannertools/tree/master/scannertools_sql): depends on [pqxx](https://github.com/jtv/libpqxx).

The [scannertools_infra](https://github.com/scanner-research/scannertools/tree/master/scannertools_infra) package contains build and test infrastructure for each of the scannertools submodules.

To install the optional dependencies that scannertools use, first clone the dependency repo and then add the source to the python pyath.

### From pip

We'll be uploading pip packages soon. In the meantime, follow the install from source direction.

### From source

First clone the repository and install the infrastructure.

```
git clone https://github.com/scanner-research/scannertools
cd scannertools
cd scannertools_infra
pip3 install --user -e .
```

Then, for each submodule you want to install, go into the the subdirectory and run pip, e.g.:

```
cd scannertools_caffe
pip3 install --user -e .
```

To build the ops with GPU compatibility, pass the path to your CUDA installation:

```
pip3 install --install-option="--build-cuda=/usr/local/cuda" --user -e .
```

## Usage

See the documentation on [scanner.run](http://scanner.run/api.html#scannertools-the-scanner-standard-library) for usage.
