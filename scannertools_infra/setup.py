from setuptools import setup

if __name__ == "__main__":
    setup(
        name='scannertools_infra',
        version='0.1.0',
        description='Scannertools infrastructure',
        url='http://github.com/scanner-research/scannertools',
        author='Will Crichton',
        author_email='wcrichto@cs.stanford.edu',
        license='Apache 2.0',
        packages=['scannertools_infra'],
        install_requires=['pytest', 'toml', 'requests', 'scipy', 'tensorflow', 'pandas'],
        zip_safe=False)
