#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 SKA South Africa
#
# This file is part of hypercube.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

import os
from setuptools import setup, find_packages
import subprocess
from setuptools._vendor.packaging import version

def get_version():
    # Versioning code here, based on
    # http://blogs.nopcode.org/brainstorm/2013/05/20/pragmatic-python-versioning-via-setuptools-and-git-tags/

    # Fetch version from git tags, and write to version.py.
    # Also, when git is not available (PyPi package), use stored version.py.
    version_py = os.path.join('hypercube', 'version.py')

    try:
        version_git = str(subprocess.check_output(['git', 'describe',
                                                   '--tags']).rstrip())
    except:
        with open(version_py, 'r') as fh:
            version_git = str(open(version_py).read().strip().split('=')[-1].replace('"',''))

    try:
        version.parse(version_git)
    except version.InvalidVersion:
        version_git = "0.0.0" # valid dev version

    version_msg = "# Do not edit this file, pipeline versioning is governed by git tags"

    with open(version_py, 'w') as fh:
        fh.write(str(version_msg) + str(os.linesep) + "__version__=\"" +
                 str(version_git) +"\"")

    return str(version_git)

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='hypercube',
    version=get_version(),
    description='Actually an n-orthotope.',
    long_description=readme(),
    url='http://github.com/ska-sa/hypercube',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
    ],
    author='Simon Perkins',
    author_email='simon.perkins@gmail.com',
    license='GPL2',
    packages=find_packages(),
    install_requires=[
        'attridict >= 0.0.8', # replacement for attrdict
        'numpy >= 1.10.1',
        'six >= 1.10.0',
        'tabulate >= 0.7.5',
    ],
    python_requires='>=3.0',
    zip_safe=True)
