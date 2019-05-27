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

import numpy
import os
import sys
import unittest

import hypercube

def print_versions():
    """
    Print the versions of software relied upon by montblanc.
    Inspired by numexpr testing suite.
    """
    print(('-=' * 38))
    print(('Python version:    %s' % sys.version))
    print(('Hypercube version: %s' % hypercube.__version__))
    print(("NumPy version:     %s" % numpy.__version__))

    if os.name == 'posix':
        (sysname, nodename, release, version, machine) = os.uname()
        print(('Platform:          %s-%s' % (sys.platform, machine)))

    print(('-=' * 38))

def suite():
    from . import test_cube

    test_suite = unittest.TestSuite()
    niter = 1

    for n in range(niter):
        test_suite.addTest(unittest.makeSuite(test_cube.Test))

    return test_suite

def test():
    print_versions()
    return unittest.TextTestRunner().run(suite())
