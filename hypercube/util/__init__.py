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

from attrdict import AttrDict
import numpy as np

from hypercube.dims import DimData

def array_bytes(array):
    """ Estimates the memory of the supplied array in bytes """
    return np.product(array.shape)*np.dtype(array.dtype).itemsize

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


def reify_dims(dims, copy=True):
    """
    Reify dimensions. If copy is True,
    returns a copy of dims else performs this inplace.
    """

    from hypercube.expressions import parse_expression

    dims = ({ k : d.copy() for k, d in dims.iteritems() }
        if copy else dims)
    G = { d.name: d.global_size for d in dims.itervalues() }
    L = { d.name: d.local_size for d in dims.itervalues() }
    E0 = { d.name: d.extents[0] for d in dims.itervalues() }
    E1 = { d.name: d.extents[1] for d in dims.itervalues() }

    for n, d in dims.iteritems():
        d[DimData.GLOBAL_SIZE] = parse_expression(d[DimData.GLOBAL_SIZE],
            variables=G, expand=True)
        d[DimData.LOCAL_SIZE] = parse_expression(d[DimData.LOCAL_SIZE],
            variables=L, expand=True)

        ext0 = parse_expression(d[DimData.EXTENTS][0],
            variables=E0, expand=True)
        ext1 = parse_expression(d[DimData.EXTENTS][1],
            variables=E1, expand=True)

        d[DimData.EXTENTS] = [ext0, ext1]

        # Force a check of the dimension constraints at this point
        d.validate()

    return dims

def reify_arrays(arrays, reified_dims, copy=True):
    """
    Reify arrays, given the supplied reified dimensions. If copy is True,
    returns a copy of arrays else performs this inplace.
    """
    arrays = ({ k : AttrDict(**a) for k, a in arrays.iteritems() }
        if copy else arrays)

    for n, a in arrays.iteritems():
        a.shape = tuple(reified_dims[v][DimData.LOCAL_SIZE]
            if isinstance(v, str) else v for v in a.shape)

    return arrays