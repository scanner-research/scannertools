from scannerpy.op import register_module
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from scannerpy import protobufs
from multiprocessing import cpu_count
import os
import subprocess as sp
import sys


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = sp.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        output_dir = os.path.join(ext.sourcedir, '..', ext.name, 'build')
        cfg = 'Debug' if self.debug else 'RelWithDebugInfo'
        build_args = ['--config', cfg, '--', '-j{}'.format(cpu_count())]
        cmake_args = [
            '-DCMAKE_INSTALL_PREFIX=' + output_dir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
            '-DCMAKE_BUILD_TYPE=' + cfg
        ]

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        sp.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp, env=env)
        sp.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp)
        sp.check_call(['make', 'install'], cwd=self.build_temp)
        print()  # Add an empty line for cleaner output


def _register_module(path):
    cwd = os.path.dirname(os.path.abspath(path))
    module = os.path.basename(cwd)
    so_path = os.path.join(cwd, 'build/lib{}.so'.format(module))
    proto_path = os.path.join(cwd, 'build/{}_pb2.py'.format(module))
    if os.path.isfile(so_path):
        register_module(so_path, proto_path if os.path.isfile(proto_path) else None)
        if os.path.isfile(proto_path):
            protobufs.add_module(proto_path)
