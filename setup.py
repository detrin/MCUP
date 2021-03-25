# -*- coding: utf-8 -*-
"""setup.py"""

import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


class Tox(TestCommand):
    user_options = [("tox-args=", "a", "Arguments to pass to tox")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.tox_args = None

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import tox
        import shlex

        if self.tox_args:
            errno = tox.cmdline(args=shlex.split(self.tox_args))
        else:
            errno = tox.cmdline(self.test_args)
        sys.exit(errno)


classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]

requires = ["setuptools", "numpy", "scipy", "numdifftools"]

extras_require = {
    "reST": ["Sphinx"],
}
if os.environ.get("READTHEDOCS", None):
    extras_require["reST"].append("recommonmark")

setup(
    name="mcup",
    version="0.1.1",
    description="MCUP will propagate uncertainty of your data points to the parameters of the regression using a Monte Carlo approach.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    keywords="physics, stats, error, uncertainty, propagation",
    author="Daniel Herman",
    author_email="daniel.herman@pm.me",
    url="https://github.com/detrin/MCUP",
    classifiers=classifiers,
    packages=["mcup"],
    data_files=[],
    install_requires=requires,
    include_package_data=True,
    extras_require=extras_require,
    tests_require=["tox"],
    cmdclass={"test": Tox},
)
