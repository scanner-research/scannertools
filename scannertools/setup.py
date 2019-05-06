from setuptools import setup
from scannertools_infra import CMakeExtension, CMakeBuild, CudaInstallCommand, CudaDevelopCommand

if __name__ == "__main__":
    setup(name='scannertools',
          version='0.2.15',
          description='Video analytics toolkit',
          url='http://github.com/scanner-research/scannertools',
          author='Will Crichton',
          author_email='wcrichto@cs.stanford.edu',
          license='Apache 2.0',
          packages=['scannertools'],
          install_requires=['torch>=1.1.0', 'scipy>=1.2.0'],
          setup_requires=['pytest-runner', 'scannertools_infra'],
          tests_require=['pytest', 'requests'],
          cmdclass=dict(build_ext=CMakeBuild,
                        install=CudaInstallCommand,
                        develop=CudaDevelopCommand),
          ext_modules=[CMakeExtension('scannertools', 'scannertools_cpp')],
          zip_safe=False)
