from setuptools import setup

setup(
    name='scannertools',
    version='0.2.15',
    description='Video analytics toolkit',
    url='http://github.com/scanner-research/scannertools',
    author='Will Crichton',
    author_email='wcrichto@cs.stanford.edu',
    license='Apache 2.0',
    packages=['scannertools'],
    install_requires=[
        'requests', 'numpy', 'scipy', 'requests', 'attrs', 'pyyaml', 'cloudpickle', 'tqdm'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    zip_safe=False)
