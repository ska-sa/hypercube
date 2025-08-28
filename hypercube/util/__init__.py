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

import attridict as AttrDict
import numpy as np

from hypercube.dims import Dimension

def array_bytes(array):
    """ Estimates the memory of the supplied array in bytes """
    return np.prod(array.shape)*np.dtype(array.dtype).itemsize

def fmt_bytes(nbytes):
    """ Returns a human readable string, given the number of bytes """
    for x in ['B','KB','MB','GB', 'TB']:
        if nbytes < 1024.0:
            return "%3.1f%s" % (nbytes, x)
        nbytes /= 1024.0

    return "%.1f%s" % (nbytes, 'PB')

def shape_name(name):
    """ Constructs a name for the array shape member, based on the array name """
    return name + '_shape'

def dtype_name(name):
    """ Constructs a name for the array data-type member, based on the array name """
    return name + '_dtype'

def setter_name(name):
    """ Constructs a name for the property, based on the property name """
    return 'set_' + name

def reify_arrays(arrays, dims, copy=True):
    """
    Reify arrays, given the supplied dimensions. If copy is True,
    returns a copy of arrays else performs this inplace.
    """
    arrays = ({ k : AttrDict(**a) for k, a in arrays.items() }
        if copy else arrays)

    for n, a in arrays.items():
        a.shape = tuple(dims[v].extent_size if isinstance(v, str) else v
            for v in a.shape)

    return arrays