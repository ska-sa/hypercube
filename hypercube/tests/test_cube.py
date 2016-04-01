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

import unittest
import sys

class Test(unittest.TestCase):
    """
    """

    def setUp(self):
        """ Set up each test case """
        pass

    def tearDown(self):
        """ Tear down each test case """
        pass

    def test_parse_expression(self):
        """ Test expression parsing """
        from hypercube.expressions import parse_expression

        # Set up our problem size
        ntime, na, nchan = 100, 64, 128
        nbl = na*(na-1)//2
        nvis = ntime*nbl*nchan

        # Check that the parser expression produces a results
        # that agrees with our manual calculation
        assert nvis == parse_expression('nvis',
            variables={ 'ntime' : ntime, 'na' : na, 'nchan': nchan,
                'nbl': 'na*(na-1)//2',
                'nvis': 'ntime*nbl*nchan' })

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
