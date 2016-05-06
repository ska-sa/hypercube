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

from __future__ import print_function

import sys

import numpy as np
import hypercube as hc
import hypercube.util as hcu

cube = hc.HyperCube()

# Register gridding dimensions
cube.register_dimension('grid_width', 10240, description='Grid Width')
cube.register_dimension('grid_height', 10240, description='Grid Height')
cube.register_dimension('nchan', 32768, description='Channels')
cube.register_dimension('nfacet', 64, description='Facets')
cube.register_dimension('facet_width', 128, description='Facet Width')
cube.register_dimension('facet_height', 128, description='Facet Height')
cube.register_dimension('npol', 4, description='Polarisations')

# Register gridding arrays in terms of dimensions above
cube.register_array('main_grid', ('grid_width', 'grid_height', 'nchan', 'npol'),
    dtype=np.complex128)
cube.register_array('facets', ('nfacet', 'facet_width', 'facet_height', 'nchan', 'npol'),
    dtype=np.complex128)

# Above produces a 50.5TB problem size,
# need to reduce the local size of our problem
print(cube)
print('\n'*4)
print('REDUCING LOCAL PROBLEM SIZE\n',
    'grid_width => 256\n',
    'grid_height => 256\n',
    'nchan => 128\n',
    '\n'
    'Note: extents[1] - extents[0] <= local_size')
print('\n'*3)

# Reduce local grid width to 1000 and globally handle 1000 - 1256
cube.update_dimension(name='grid_width',
    local_size=256, extents=[1000, 1256],
    safety=False) # Turn off the safety for local size updates

# Reduce local grid height to 1000 and globally handle range 1000 - 1256
cube.update_dimension(name='grid_height',
    local_size=256, extents=[1000, 1256],
    safety=False) # Turn off the safety for local size updates

# Reduce local number of channels to 128 and globally handle range 600-700
cube.update_dimension(name='nchan',
    local_size=128, extents=[600, 700],
    safety=False) # Turn off the safety for local size updates

print (cube)

# Create a hypercube, which will actually create
# numpy arrays on the hypercube corresponding to local array sizes
np_cube = hc.HyperCube()

# Register dimension and array information from the original hypercube
np_cube.register_dimensions([d for d in cube.dimensions().itervalues()])
np_cube.register_arrays([a for a in cube.arrays().itervalues()])
hc.create_local_numpy_arrays_on_cube(np_cube)

# Get some dimension information to check our numpy shape size
grid_width, grid_height, nchan, npol = np_cube.dim_local_size(
    'grid_width', 'grid_height', 'nchan', 'npol')
assert np_cube.main_grid.shape == (grid_width, grid_height, nchan, npol)

# Do some Numpy-like things on channel 1
np_cube.main_grid[:,:,1] = 1 - 1*1j

print(hcu.fmt_bytes(np_cube.main_grid.nbytes))

try:
    import pycuda.autoinit

    # Create a cuda hypercube, which will hold gpu arrays
    # corresponding to local array sizes
    cuda_cube = hc.HyperCube()

    # Register dimension and array information from the original hypercube
    cuda_cube.register_dimensions([d for d in cube.dimensions().itervalues()])
    cuda_cube.register_arrays([a for a in cube.arrays().itervalues()])

    # Reduce local number of channels to 128 and globally handle range 600-700
    cuda_cube.update_dimension(name='nfacet',
        local_size=1, extents=[0, 1],
        safety=False) # Turn off the safety for local size updates

    hc.create_local_pycuda_arrays_on_cube(cuda_cube)

    print(cuda_cube)
except:
    raise ValueError("PyCUDA not installed or not "
        "enough GPU memory to hold {b}"
            .format(b=np_cube.mem_required())), None, sys.exc_info()[2]

