from setuptools import setup

setup(
    name='tixelbox',
    version='0.1',
    description='Video analytics toolkit',
    url='http://github.com/scanner-research/tixelbox',
    author='Alex Poms and Will Crichton',
    author_email='apoms@cs.stanford.edu',
    license='Apache 2.0',
    packages=['tixelbox'],
    install_requires=['requests', 'numpy', 'scipy', 'requests'],
    zip_safe=False)
