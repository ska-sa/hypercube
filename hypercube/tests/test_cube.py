#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2016 SKA South Africa
#
# This file is part of hc.
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

import numpy as np

import hypercube as hc

class Test(unittest.TestCase):
    """
    """

    def setUp(self):
        """ Set up each test case """
        pass

    def tearDown(self):
        """ Tear down each test case """
        pass

    def test_dimension_registration_and_reification(self):
        """ Test dimension registration and reification """
        # Set up our problem size
        ntime, na, nchan, npol = 100, 64, 128, 4
        nvis = ntime*nchan*na*(na-1)//2

        nbl_expr = 'na*(na-1)//2'
        nvis_expr = 'ntime*nbl*nchan'

        # Set up the hypercube dimensions
        cube = hc.HyperCube()
        cube.register_dimension('ntime', ntime)
        cube.register_dimension('na', na)
        cube.register_dimension('nchan', nchan)
        cube.register_dimension('npol', npol)
        cube.register_dimension('nbl', nbl_expr)
        cube.register_dimension('nvis', nvis_expr)

        # Test that we still have an abstract dimensions when
        # no reification is requested
        dims = cube.dimensions()
        self.assertTrue(dims['nvis'].global_size == nvis_expr)
        self.assertTrue(dims['nvis'].local_size == nvis_expr)
        self.assertTrue(dims['nvis'].lower_extent == nvis_expr)
        self.assertTrue(dims['nvis'].upper_extent == nvis_expr)

        # Test that we now have concrete dimensions when
        # reification is requested
        dims = cube.dimensions(reify=True)
        self.assertTrue(dims['nvis'].global_size == nvis)
        self.assertTrue(dims['nvis'].local_size == nvis)
        self.assertTrue(dims['nvis'].lower_extent == 0)
        self.assertTrue(dims['nvis'].upper_extent == nvis)

        # Reduce the local size of the ntime, na and nchan dimensions
        local_ntime = ntime//2
        local_na = na - 2
        local_nchan = nchan // 4
        local_nvis = local_ntime*local_nchan*local_na*(local_na-1)//2

        cube.update_dimension(name='ntime', local_size=local_ntime,
            lower_extent=0, upper_extent=local_ntime)
        cube.update_dimension(name='na', local_size=local_na,
            lower_extent=0, upper_extent=local_na)
        cube.update_dimension(name='nchan', local_size=local_nchan,
            lower_extent=0, upper_extent=local_nchan)

        # Test that we now have concrete dimensions when
        # reification is requested
        dims = cube.dimensions(reify=True)
        self.assertTrue(dims['nvis'].global_size == nvis)
        self.assertTrue(dims['nvis'].local_size == local_nvis)
        self.assertTrue(dims['nvis'].lower_extent == 0)
        self.assertTrue(dims['nvis'].upper_extent == local_nvis)

        # Test individual dimension retrieval
        dim = cube.dimension('nvis')
        self.assertTrue(dim.global_size == nvis_expr)
        self.assertTrue(dim.local_size == nvis_expr)
        self.assertTrue(dim.lower_extent == nvis_expr)
        self.assertTrue(dim.upper_extent == nvis_expr)

        # Test individual dimension reification
        dim = cube.dimension('nvis', reify=True)
        self.assertTrue(dim.global_size == nvis)
        self.assertTrue(dim.local_size == local_nvis)
        self.assertTrue(dim.lower_extent == 0)
        self.assertTrue(dim.upper_extent == local_nvis)

        # Test that we still have an abstract dimensions when
        # no reification is requested
        dims = cube.dimensions()
        self.assertTrue(dims['nvis'].global_size == nvis_expr)
        self.assertTrue(dims['nvis'].local_size == nvis_expr)
        self.assertTrue(dims['nvis'].lower_extent == nvis_expr)
        self.assertTrue(dims['nvis'].upper_extent == nvis_expr)

    def test_dimension_updates(self):
        """ Test dimension updates """
        # Set up our problem size
        ntime, na, nchan = 100, 64, 128

        # Set up the hypercube dimensions
        cube = hc.HyperCube()
        cube.register_dimension('ntime', ntime)
        cube.register_dimension('na', na)
        cube.register_dimension('nchan', nchan)

        tdim = cube.dimensions()['ntime']
        self.assertTrue(tdim.global_size == tdim.local_size)

        # Check that setting the global size greater than the local size succeeds
        cube.update_dimension(name='ntime', global_size=120)
        self.assertTrue(tdim.global_size == 120)

        # Check that setting the global size less than the local size fails
        with self.assertRaises(ValueError) as cm:
            cube.update_dimension(name='ntime', global_size=80)

        self.assertTrue('local size {nt}'.format(nt=ntime) in cm.exception.message)
        self.assertTrue('global size {gs}'.format(gs=80) in cm.exception.message)

        # Check that setting the global_size and local_size less than the upper_extent fails
        with self.assertRaises(ValueError) as cm:
            cube.update_dimension(name='ntime', global_size=80, local_size=80)

        # This should succeed
        cube.update_dimension(name='ntime', global_size=80, local_size=80, upper_extent=80)

    def test_array_registration_and_reification(self):
        """ Test array registration and reification """
        # Set up our problem size
        ntime, na, nchan, npol = 100, 64, 128, 4

        # Set up the hypercube dimensions
        cube = hc.HyperCube()
        cube.register_dimension('ntime', ntime)
        cube.register_dimension('na', na)
        cube.register_dimension('nchan', nchan)
        cube.register_dimension('npol', npol)
        cube.register_dimension('nbl', 'na*(na-1)//2')
        cube.register_dimension('nvis', 'ntime*nbl*nchan')

        # Register the visibility array with abstract shape
        VIS = 'visibilities'
        abstract_shape = ('ntime','nbl','nchan','npol')
        cube.register_array(VIS, abstract_shape, np.complex128)

        # Test that we still have an abstract shape when
        # no reification is requested
        arrays = cube.arrays()
        self.assertTrue(arrays[VIS].shape == abstract_shape)

        # Test that we have a concrete shape after reifying the arrays
        arrays = cube.arrays(reify=True)
        concrete_shape = (ntime, na*(na-1)//2, nchan, npol)
        self.assertTrue(arrays[VIS].shape == concrete_shape)

        # Update the local size and extents of the time dimension
        local_ntime = ntime//2
        cube.update_dimension(name='ntime', local_size=local_ntime,
            lower_extent=0, upper_extent=local_ntime)

        # Test that the concrete shape reflects the new local_size
        # after reifying the arrays
        arrays = cube.arrays(reify=True)
        concrete_shape = (local_ntime, na*(na-1)//2, nchan, npol)
        self.assertTrue(arrays[VIS].shape == concrete_shape)

        # Test individual array retrieval
        array = cube.array(VIS)
        self.assertTrue(array.shape == abstract_shape)

        # Test individual array reification
        array = cube.array(VIS, reify=True)
        self.assertTrue(array.shape == concrete_shape)

        # Test that we still have an abstract shape when
        # no reification is requested
        arrays = cube.arrays()
        self.assertTrue(arrays[VIS].shape == abstract_shape)

    def test_array_creation(self):
        ntime, na, nchan, npol = 100, 64, 128, 4
        nbl = na*(na-1)//2

        # Set up the hypercube dimensions
        cube = hc.HyperCube()
        cube.register_dimension('ntime', ntime)
        cube.register_dimension('na', na)
        cube.register_dimension('nchan', nchan)
        cube.register_dimension('npol', npol)
        cube.register_dimension('nbl', 'na*(na-1)//2')
        cube.register_dimension('nvis', 'ntime*nbl*nchan')

        # Register the array with abstract shapes
        cube.register_array('visibilities', ('ntime','nbl','nchan','npol'), np.complex128)
        cube.register_array('uvw', ('ntime', 'nbl', 3), np.float64)
        cube.register_array('ant_pairs', (2, 'ntime', 'nbl'), np.int64)

        # Create the arrays
        arrays = hc.create_local_arrays(cube.arrays(reify=True))

        # Check that we get numpy arrays by default
        for a in arrays.itervalues():
            self.assertTrue(isinstance(a, np.ndarray))

        # Check that the shape is correct
        self.assertTrue(arrays['visibilities'].shape == (ntime, nbl, nchan, 4))
        self.assertTrue(arrays['uvw'].shape == (ntime, nbl, 3))
        self.assertTrue(arrays['ant_pairs'].shape == (2, ntime, nbl))

        # Check that the type is correct
        self.assertTrue(arrays['visibilities'].dtype == np.complex128)
        self.assertTrue(arrays['uvw'].dtype == np.float64)
        self.assertTrue(arrays['ant_pairs'].dtype == np.int64)

        # Create the arrays
        arrays = hc.create_local_numpy_arrays_on_cube(cube)

        # Check that we get numpy arrays by default
        for a in arrays.itervalues():
            self.assertTrue(isinstance(a, np.ndarray))

        # Check that the shape is correct
        self.assertTrue(cube.visibilities.shape == (ntime, nbl, nchan, 4))
        self.assertTrue(cube.uvw.shape == (ntime, nbl, 3))
        self.assertTrue(cube.ant_pairs.shape == (2, ntime, nbl))

        # Check that the type is correct
        self.assertTrue(cube.visibilities.dtype == np.complex128)
        self.assertTrue(cube.uvw.dtype == np.float64)
        self.assertTrue(cube.ant_pairs.dtype == np.int64)

    def test_dim_queries(self):
        # Set up our problem size
        ntime, na, nchan = 100, 64, 128
        nbl = na*(na-1)//2
        nvis = ntime*nbl*nchan

        # Create a cube and register some dimensions
        cube = hc.HyperCube()
        cube.register_dimension('ntime', ntime, lower_extent=1)
        cube.register_dimension('na', na, lower_extent=2)
        cube.register_dimension('nchan', nchan, lower_extent=3)
        cube.register_dimension('nbl', nbl, lower_extent=4)
        cube.register_dimension('nvis', nvis, lower_extent=5)

        args = ['ntime','na','nbl','nchan','nvis']

        # Test that the mutiple argument form works
        _ntime, _na, _nbl, _nchan, _nvis = cube.dim_global_size(*args)

        self.assertTrue(_ntime == ntime)
        self.assertTrue(_nbl == nbl)
        self.assertTrue(_na == na)
        self.assertTrue(_nchan == nchan)
        self.assertTrue(_nvis == nvis)

        # Test that the multiple argument form works
        ((tl, tu), (al, au), (bl, bu),
            (cl, cu), (vl, vu)) = cube.dim_extents(*args)

        self.assertTrue(tl == 1 and tu == ntime)
        self.assertTrue(al == 2 and au == na)
        self.assertTrue(cl == 3 and cu == nchan)
        self.assertTrue(bl == 4 and bu == nbl)
        self.assertTrue(vl == 5 and vu == nvis)

        # Test that singleton argument form works
        tl, tu = cube.dim_extents('ntime')

        self.assertTrue(tl == 1 and tu == ntime)

        # Test that the mutiple argument form works
        _ntime, _na, _nbl, _nchan, _nvis = cube.dim_extent_size(*args)

        self.assertTrue(_ntime == tu - tl)
        self.assertTrue(_nbl == bu - bl)
        self.assertTrue(_na == au - al)
        self.assertTrue(_nchan == cu - cl)
        self.assertTrue(_nvis == vu - vl)

        # Test that singleton argument form works
        _ntime = cube.dim_extent_size('ntime')
        self.assertTrue(_ntime == tu - tl)

        # Test that the multiple arguments packed into a string form works
        _ntime, _na, _nbl, _nchan, _nvis = cube.dim_global_size(','.join(args))

        self.assertTrue(_ntime == ntime)
        self.assertTrue(_nbl == nbl)
        self.assertTrue(_na == na)
        self.assertTrue(_nchan == nchan)
        self.assertTrue(_nvis == nvis)

        local_ntime, local_na, local_nchan = 10, 7, 16
        local_nbl = local_na*(local_na-1)//2
        local_nvis = local_ntime*local_nbl*local_nchan

        values = [local_ntime, local_na, local_nbl, local_nchan, local_nvis]

        for arg, ls in zip(args, values):
            cube.update_dimension(name=arg, local_size=ls,
                lower_extent=0, upper_extent=ls)

        # Test that the mutiple argument form works
        _ntime, _na, _nbl, _nchan, _nvis = cube.dim_local_size(*args)

        self.assertTrue(_ntime == local_ntime)
        self.assertTrue(_nbl == local_nbl)
        self.assertTrue(_na == local_na)
        self.assertTrue(_nchan == local_nchan)
        self.assertTrue(_nvis == local_nvis)

    def test_ignore_extents(self):
        # Set up our problem size
        ntime, na, nchan, npol = 100, 64, 128, 4
        nbl_expr = 'na*(na-1)//2'
        nbl = na*(na-1)//2
        nvis = ntime*nbl*nchan

        # Set up the hypercube dimensions
        cube = hc.HyperCube()
        cube.register_dimension('ntime', ntime)
        cube.register_dimension('na', na)
        cube.register_dimension('nchan', nchan)
        cube.register_dimension('npol', npol)
        #import pdb; pdb.set_trace()
        cube.register_dimension('nbl', nbl_expr, ignore_extents=False)
        cube.register_dimension('nvis', 'ntime*nbl*nchan')

        dims = cube.dimensions()

        # Not using an expression, should be False
        self.assertTrue(dims['nbl'].ignore_extents == False)
        # Explicitly mandated to be False
        self.assertTrue(dims['nbl'].ignore_extents == False)
        # Is an expression, ignore_extents defaults to True
        self.assertTrue(dims['nvis'].ignore_extents == True)

        # Test that trying to set the upper_extent beyond
        # the global size fails in the case of nbl
        with self.assertRaises(ValueError) as cm:
            cube.update_dimension('nbl', upper_extent=nbl+3)
            cube.dimensions(reify=True)

        self.assertTrue('nbl' in cm.exception.message)
        self.assertTrue(str(nbl+3) in cm.exception.message)
        self.assertTrue(str(nbl) in cm.exception.message)

        # Fix up the upper_extent for the next test case
        cube.update_dimension('nbl', upper_extent=nbl_expr)

        # Test that setting the upper_extent beyond the
        # global size succeeds in the case of nvis
        cube.update_dimension('nvis', upper_extent=nvis+3)
        cube.dimensions(reify=True)

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

        # Now set up the above example on the hypercube
        cube = hc.HyperCube()
        cube.register_dimension('ntime', ntime)
        cube.register_dimension('na', na)
        cube.register_dimension('nchan', nchan)
        cube.register_dimension('nbl', 'na*(na-1)//2')
        cube.register_dimension('nvis', 'ntime*nbl*nchan')

        # Sanity check our global dimension sizes
        G = cube.dim_global_size_dict()
        self.assertTrue(G['nbl'] == nbl)
        self.assertTrue(G['nvis'] == nvis)

        # Create some local variable sizes
        local_ntime, local_na, local_nchan = 10, 7, 16
        local_nbl = local_na*(local_na-1)//2
        local_nvis = local_ntime*local_nbl*local_nchan

        local_dims = ['ntime', 'na', 'nchan']
        values = [local_ntime, local_na, local_nchan]

        # Update local size of selected dimensions
        for arg, ls in zip(local_dims, values):
            cube.update_dimension(name=arg, local_size=ls,
                lower_extent=0, upper_extent=ls)

        # Check that local dimension query produces
        # the corrected derived sizes
        L = cube.dim_local_size_dict()
        self.assertTrue(L['nbl'] == local_nbl)
        self.assertTrue(L['nvis'] == local_nvis)

        # Sanity check our global dimension sizes
        G = cube.dim_global_size_dict()
        self.assertTrue(G['nbl'] == nbl)
        self.assertTrue(G['nvis'] == nvis)


    def test_parse_expression_fail(self):
        """ Test expression parsing failure """
        from hypercube.expressions import parse_expression

        # Check that a missing nchan in nvis produces an exception
        with self.assertRaises(ValueError) as cm:
            parse_expression('nvis',
                variables={ 'ntime': 10, 'nbl': 21,
                    'nvis': 'ntime*nbl*nchan' })

            self.assertTrue("Unable to evaluate "
                "expression 'ntime*nbl*nchan'" in str(cm.exception))

            self.assertTrue("as variable 'nchan' was not"
                " in the variable dictionary.")

    def test_iterators(self):
        """ Test chunk iteration """
        # Set up our problem size
        ntime, na, nchan = 100, 64, 128
        nbl = na*(na-1)//2
        nvis = ntime*nbl*nchan

        # Create a cube and register some dimensions
        cube = hc.HyperCube()
        cube.register_dimension('ntime', ntime)
        cube.register_dimension('na', na)
        cube.register_dimension('nchan', nchan)
        cube.register_dimension('nbl', nbl)
        cube.register_dimension('nvis', nvis)
        cube.register_array('uvw', ('ntime', 'nbl', 3), np.float64)

        tsize, asize = 9, 5

        #=============================
        # Global Scope Iteration Tests
        #=============================

        # Create array to iterate over
        A = np.arange(ntime*na).reshape(ntime, na)
        A_sum = A.sum()

        # Test that iterating over offsets works
        S = sum(A[ts:te,as_:ae].sum() for (ts, te), (as_, ae) in
            cube.endpoint_iter(('ntime', tsize), ('na', asize),
            scope='global_size'))
        self.assertTrue(S == A_sum)

        # Test that iterating over tuple indices works
        S = sum(A[i].sum() for i in
            cube.slice_iter(('ntime', tsize), ('na', asize),
            scope='global_size'))
        self.assertTrue(S == A_sum)

        # Test that iterating over destructured tuple indices works
        S = sum(A[t,a].sum() for t, a in
            cube.slice_iter(('ntime', tsize), ('na', asize),
            scope='global_size'))
        self.assertTrue(S == A_sum)

        # Test that iterating over hypercubes works
        S = sum(A[c.slice_index('ntime', 'na')].sum() for c in
            cube.cube_iter(('ntime', tsize), ('na', asize),
            scope='global_size'))
        self.assertTrue(S == A_sum)

        # Test that arrays aren't copied over by default
        for c in cube.cube_iter(('ntime', tsize), ('na', asize),
            scope='global_size'):
            self.assertTrue('uvw' not in c.arrays().keys())
            break

        # Test that arrays aren't copied over when requested
        for c in cube.cube_iter(('ntime', tsize), ('na', asize),
            scope='global_size', arrays=True):
            self.assertTrue('uvw' in c.arrays().keys())
            break

        # Test that iterating over dimension dictionaries works
        S = 0
        for d in cube.dim_iter(('ntime', tsize), ('na', asize)):
            cube.update_dimensions(d)
            S += A[cube.slice_index('ntime', 'na')].sum()

        self.assertTrue(S == A_sum)

        #=============================
        # Local Scope Iteration Tests
        #=============================

        ltime, lna = 20, 27

        # Test that we can iterate over a dimension's local size
        cube.update_dimension(name='ntime', global_size=ntime,
            local_size=ltime, lower_extent=0, upper_extent=ltime)

        cube.update_dimension(name='na', global_size=na,
            local_size=lna, lower_extent=0, upper_extent=lna)

        # Sum over a portion of the array now!
        A_sum = A[0:ltime,0:lna].sum()

        # Test that iterating over offsets works
        S = sum(A[ts:te,as_:ae].sum() for (ts, te), (as_, ae) in
            cube.endpoint_iter(('ntime', tsize), ('na', asize)))
        self.assertTrue(S == A_sum)

        # Test that iterating over tuple indices works
        S = sum(A[i].sum() for i in
            cube.slice_iter(('ntime', tsize), ('na', asize)))
        self.assertTrue(S == A_sum)

        # Test that iterating over destructured tuple indices works
        S = sum(A[t,a].sum() for t, a in
            cube.slice_iter(('ntime', tsize), ('na', asize)))
        self.assertTrue(S == A_sum)

        # Test that iterating over hypercubes works
        S = sum(A[c.slice_index('ntime', 'na')].sum() for c in
            cube.cube_iter(('ntime', tsize), ('na', asize)))
        self.assertTrue(S == A_sum)

        # Test that iterating over dimension dictionaries works
        S = 0
        for d in cube.dim_iter(('ntime', tsize), ('na', asize)):
            cube.update_dimensions(d)
            S += A[cube.slice_index('ntime', 'na')].sum()

        self.assertTrue(S == A_sum)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
