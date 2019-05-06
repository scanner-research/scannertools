from setuptools import setup
from scannertools_infra import CMakeExtension, CMakeBuild, CudaInstallCommand, CudaDevelopCommand

if __name__ == "__main__":
    setup(name='scannertools_sql',
          version='0.2.15',
          description='Video analytics toolkit',
          url='http://github.com/scanner-research/scannertools',
          author='Will Crichton',
          author_email='wcrichto@cs.stanford.edu',
          license='Apache 2.0',
          packages=['scannertools_sql'],
          install_requires=[],
          setup_requires=['pytest-runner', 'scannertools_infra'],
          tests_require=[
              'pytest', 'psycopg2-binary == 2.7.6.1', 'testing.postgresql == 1.3.0',
              'scannertools_infra'
          ],
          cmdclass=dict(build_ext=CMakeBuild,
                        install=CudaInstallCommand,
                        develop=CudaDevelopCommand),
          ext_modules=[CMakeExtension('scannertools_sql', 'scannertools_sql_cpp')],
          zip_safe=False)
