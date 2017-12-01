import sys
from setuptools.command.test import test as TestCommand
from setuptools import setup, find_packages
from pip.req import parse_requirements
from pip.download import PipSession

install_reqs = parse_requirements('requirements.txt', session=PipSession())
reqs = [str(ir.req) for ir in install_reqs]


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name='statoil',
    version='0.1',
    description='',
    author='Tomáš Přinda',
    author_email='tomas.prinda@gmail.com',
    url='',
    packages=find_packages(),
    scripts=[],
    setup_requires=['pytest-runner'],
    install_requires=reqs,
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    include_package_data=True,
    package_data={
        # Add all files under data/ directory, which is in empty_package_skeleton module
        # This data will be part of this package.
        # Access them with pkg_resources module.
        # Folder with data HAVE TO be in some module, so dont add it to folder with tests, which SHOULD NOT be a module.
    },
)
