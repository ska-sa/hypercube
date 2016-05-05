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

# Register time, baseline, channel and polarisation dimensions
cube.register_dimension('ntime', 1000, description='Timesteps')
cube.register_dimension('na', 64, description='Antenna')
cube.register_dimension('nchan', 32768, description='Channels')
cube.register_dimension('npol', 4, description='Polarisations')
cube.register_dimension('nsrc', 100, description='Sources')
# Note these dimension sizes are described in terms of other dimensions
cube.register_dimension('nbl', 'na*(na-1)//2', description='Baselines')
cube.register_dimension('nvis', 'ntime*nbl*nchan', description='Visibilities')

# Register visibility and UVW arrays
cube.register_array('lm', ('nsrc', 2), dtype=np.float32)
cube.register_array('visibilities', ('nvis', 'npol'), dtype=np.complex128)
cube.register_array('flag', ('ntime', 'nbl', 'nchan', 'npol'), dtype=np.int32)
cube.register_array('weight', ('ntime', 'nbl', 'nchan', 'npol'), dtype=np.float64)
cube.register_array('uvw', ('ntime', 'nbl', 3), dtype=np.complex128)

print cube
