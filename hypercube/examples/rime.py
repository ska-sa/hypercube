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
from hypercube import hypercube

cube = hypercube('hypercube')

# Register time, baseline, channel and polarisation dimensions
cube.register_dimension('ntime', 1000, description='Number of timesteps')
cube.register_dimension('nbl', 64*65//2, description='Number of baselines')
cube.register_dimension('nchan', 32768, description='Number of channels')
cube.register_dimension('npol', 4, description='Number of polarisations')
cube.register_dimension('nsrc', 100, description='Number of sources')

# Register visibility and UVW arrays
cube.register_array('lm', ('nsrc', 2), dtype=np.float32)
cube.register_array('visibilities', ('ntime', 'nbl', 'nchan', 'npol'), dtype=np.complex128)
cube.register_array('flag', ('ntime', 'nbl', 'nchan', 'npol'), dtype=np.int32)
cube.register_array('weight', ('ntime', 'nbl', 'nchan', 'npol'), dtype=np.float64)
cube.register_array('uvw', ('ntime', 'nbl', 3), dtype=np.complex128)

print cube
