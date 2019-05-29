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

import numpy as np
import hypercube as hc

cube = hc.HyperCube()

ntime = 1000
na = 64
nchan = 32768
nbl = 64*(64-1)//2
nvis = ntime*nbl*nchan

# Register time, baseline, channel and polarisation dimensions
cube.register_dimension('ntime', ntime, description='Timesteps')
cube.register_dimension('na', na, description='Antenna')
cube.register_dimension('nchan', nchan, description='Channels')
cube.register_dimension('npol', 4, description='Polarisations')
cube.register_dimension('nsrc', 100, description='Sources')
cube.register_dimension('nbl', nbl, description='Baselines')
cube.register_dimension('nvis', nvis, description='Visibilities')

# Register visibility and UVW arrays
cube.register_array('lm', ('nsrc', 2), dtype=np.float32)
cube.register_array('visibilities', ('ntime', 'nbl', 'nchan', 'npol'), dtype=np.complex128)
cube.register_array('flag', ('ntime', 'nbl', 'nchan', 'npol'), dtype=np.int32)
cube.register_array('weight', ('ntime', 'nbl', 'nchan', 'npol'), dtype=np.float64)
cube.register_array('uvw', ('ntime', 'nbl', 3), dtype=np.complex128)

# Iterate over time, baseline and channel
# This is like a 3 level for loop
iter_dims = ('ntime', 'nbl', 'nchan')
iter_strides = (100, 24, 64)
iter_args = list(zip(iter_dims, iter_strides))

# Do the iteration, updating the cube with extent
# and chunk size information
for i, d in enumerate(cube.dim_iter(*iter_args)):
    cube.update_dimensions(d)
    print('extents', list(zip(iter_dims, cube.dim_extents(*iter_dims))))
    # Query the time dimension
    print('ntime dimension', cube.dimension('ntime'))
    # Query the shape of the visibilities array.
    # First version gives the abstract shape (defined above)
    # Second reifies it with dimension information
    print('abstract visibilities shape', cube.array('visibilities').shape)
    print('reified visibilities shape', cube.array('visibilities', reify=True).shape)

    # Expanded version of cube.dim_extents
    #print zip(iter_dims, cube.dim_lower_extent(*iter_dims))
    #print zip(iter_dims, cube.dim_upper_extent(*iter_dims))

    # These won't change during the loop, except local size
    # at the end
    #print zip(iter_dims, cube.dim_local_size(*iter_dims))
    #print zip(iter_dims, cube.dim_global_size(*iter_dims))


print(cube)
