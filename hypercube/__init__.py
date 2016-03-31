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

from hypercube.base_cube import HyperCube
from hypercube.numpy_cube import NumpyHyperCube
from hypercube.cuda_cube import CUDAHyperCube

def hypercube(cube_type, **kwargs):
    if cube_type == 'hypercube':
        return HyperCube()
    elif cube_type == 'numpy_cube':
        return NumpyHyperCube()
    elif cube_type == 'cuda_cube':
        import pycuda.autoinit
        return CUDAHyperCube(context=pycuda.autoinit.context)
