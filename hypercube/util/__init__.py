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

from hypercube.dims import Dimension
from hypercube.expressions import parse_expression as pe

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


def reify_dims(dims):
    """
    Returns a reified copy of dims
    """

    # create variable dictionaries for each attribute
    G = { d.name: d.global_size for d in dims.itervalues() }
    L = { d.name: d.local_size for d in dims.itervalues() }
    EL = { d.name: d.lower_extent for d in dims.itervalues() }
    EU = { d.name: d.upper_extent for d in dims.itervalues() }

    # Produce a dictionary of reified dimensions
    rdims = { d.name : Dimension(name=d.name,
        global_size=pe(d.global_size, variables=G, expand=True),
        local_size=pe(d.local_size, variables=L, expand=True),
        lower_extent=pe(d.lower_extent, variables=EL, expand=True),
        upper_extent=pe(d.upper_extent, variables=EU, expand=True),
        description=d.description,
        ignore_extents=d.ignore_extents,
        zero_valid=d.zero_valid)
            for d in dims.itervalues() }

    # Validate reified dimensions
    for d in rdims.itervalues():
        d.validate()

    return rdims

def reify_arrays(arrays, reified_dims, copy=True):
    """
    Reify arrays, given the supplied reified dimensions. If copy is True,
    returns a copy of arrays else performs this inplace.
    """
    arrays = ({ k : AttrDict(**a) for k, a in arrays.iteritems() }
        if copy else arrays)

    for n, a in arrays.iteritems():
        a.shape = tuple(reified_dims[v].local_size
            if isinstance(v, str) else v for v in a.shape)

    return arrays