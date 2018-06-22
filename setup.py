from setuptools import setup

setup(
    name='scannertools',
    version='0.2.8',
    description='Video analytics toolkit',
    url='http://github.com/scanner-research/scannertools',
    author='Alex Poms and Will Crichton',
    author_email='apoms@cs.stanford.edu',
    license='Apache 2.0',
    packages=['scannertools', 'scannertools.kernels'],
    install_requires=['requests', 'numpy', 'scipy', 'requests'],
    zip_safe=False)
