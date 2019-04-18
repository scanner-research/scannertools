# scannertools: core standard library for Scanner

See [scanner.run](http://scanner.run/api/scannertools.html) for a description of the operations in this library.

## Installation

### Dependencies

Requirements:
* OpenCV >= 3.4.0
* pybind11 >= 2.2.3
* ffmeg >= 3.3.1

#### Ubuntu 16.04

```
sudo apt-get install -y ffmpeg pybind11-dev
```

See the [OpenCV docs](https://docs.opencv.org/3.4.1/d2/de6/tutorial_py_setup_in_ubuntu.html) for installing OpenCV.

#### OS X

```
brew install ffmpeg opencv@3 pybind11
```

### Library

#### From pypi

Coming soon, build from source for now.

#### From source

```
git clone https://github.com/scanner-research/scannertools
cd scannertools/scannertools_infra
pip3 install -e .
cd ../scannertools
pip3 install -e .
```

To build with GPU support, instead run:

```
pip3 install --install-option="--build-cuda=/usr/local/cuda" -e .
```
