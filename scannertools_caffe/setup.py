from setuptools import setup
from scannertools_infra import CMakeExtension, CMakeBuild, CudaInstallCommand, CudaDevelopCommand

if __name__ == "__main__":
    setup(name='scannertools_caffe',
          version='0.2.15',
          description='Video analytics toolkit',
          url='http://github.com/scanner-research/scannertools',
          author='Will Crichton',
          author_email='wcrichto@cs.stanford.edu',
          license='Apache 2.0',
          packages=['scannertools_caffe'],
          install_requires=[],
          setup_requires=['pytest-runner', 'scannertools_infra'],
          tests_require=['pytest', 'scannertools_infra'],
          cmdclass=dict(build_ext=CMakeBuild,
                        install=CudaInstallCommand,
                        develop=CudaDevelopCommand),
          ext_modules=[CMakeExtension('scannertools_caffe', 'scannertools_caffe_cpp')],
          zip_safe=False)
