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

DEFAULT_DESCRIPTION = 'The FOURTH dimension!'

class DimData(object):
    NAME = 'name'
    DESCRIPTION = 'description'
    GLOBAL_SIZE = 'global_size'
    LOCAL_SIZE = 'local_size'
    EXTENTS = 'extents'
    SAFETY = 'safety'
    ZERO_VALID = 'zero_valid'

    ALL = frozenset([NAME, DESCRIPTION, GLOBAL_SIZE, LOCAL_SIZE,
        EXTENTS, SAFETY, ZERO_VALID])

def create_dim_data(name, dim_data, **kwargs):
    return Dimension(name, dim_data, **kwargs)

class Dimension(AttrDict):
    def __init__(self, name, dim_data, **kwargs):
        """
        Create a dimension data dictionary from dim_data
        and keyword arguments. Keyword arguments will be
        used to update the dictionary.

        Arguments
        ---------
            name : str
                Name of the dimension
            dim_data : integer or another Dimension
                If integer a fresh Dimension will be created, otherwise
                dim_data will be copied.

        Keyword Arguments
        -----------------
            Any keyword arguments applicable to the Dimension
            object will be applied at the end of the construction process.
        """
        super(Dimension, self).__init__()

        # If dim_data is an integer or string,
        # start constructing a dictionary from it
        if isinstance(dim_data, (int, long, np.integer, str)):
            self[DimData.NAME] = name
            self[DimData.GLOBAL_SIZE] = dim_data
            self[DimData.LOCAL_SIZE] = dim_data
            
            if isinstance(dim_data, str):
                self[DimData.EXTENTS] = [dim_data, dim_data]
            else:
                self[DimData.EXTENTS] = [0, dim_data]

            self[DimData.DESCRIPTION] = DEFAULT_DESCRIPTION
            self[DimData.SAFETY] = True
            self[DimData.ZERO_VALID] = False
        # Otherwise directly copy the entries
        elif isinstance(dim_data, Dimension):
            for k, v in dim_data.iteritems():
                self[k] = v
            self[DimData.NAME] = name
        else:
            raise TypeError(("dim_data must be an integer or a Dimension. "
                "Received a {t} instead.").format(t=type(dim_data)))

        # Intersect the keyword arguments with dictionary values
        kwargs = {k: kwargs[k] for k in kwargs.iterkeys() if k in DimData.ALL}

        # Now update the dimension data from any keyword arguments
        self.update(kwargs)

    def copy(self):
        """ Defer to the constructor for copy operations """
        return Dimension(self[DimData.NAME], self)

    def update(self, other=None, **kwargs):
        """
        Sanitised dimension data update
        """

        from collections import Mapping, Sequence

        # Just pack everything from other into kwargs
        # for the updates below
        # See http://stackoverflow.com/a/30242574
        if other is not None:
            for k, v in other.iteritems() if isinstance(other, Mapping) else other:
                kwargs[k] = v

        if DimData.NAME in kwargs:
            self[DimData.NAME] = kwargs[DimData.NAME]

        name = self[DimData.NAME]

        if DimData.DESCRIPTION in kwargs:
            self[DimData.DESCRIPTION] = kwargs[DimData.DESCRIPTION]

        # Update options if present
        if DimData.SAFETY in kwargs:
            self[DimData.SAFETY] = kwargs[DimData.SAFETY]

        if DimData.ZERO_VALID in kwargs:
            self[DimData.ZERO_VALID] = kwargs[DimData.ZERO_VALID]

        if DimData.LOCAL_SIZE in kwargs:
            if self[DimData.SAFETY] is True:
                raise ValueError(("Modifying local size of dimension '{d}' "
                    "is not allowed by default. If you are sure you want "
                    "to do this add a '{s}' : 'False' entry to the "
                    "update dictionary.").format(d=name, s=DimData.SAFETY))

            if self[DimData.ZERO_VALID] is False and kwargs[DimData.LOCAL_SIZE] == 0:
                raise ValueError(("Modifying local size of dimension '{d}' "
                    "to zero is not valid. If you are sure you want "
                    "to do this add a '{s}' : 'True' entry to the "
                    "update dictionary.").format(d=name, s=DimData.ZERO_VALID))

            self[DimData.LOCAL_SIZE] = kwargs[DimData.LOCAL_SIZE]

        if DimData.EXTENTS in kwargs:
            exts = kwargs[DimData.EXTENTS]
            if (not isinstance(exts, Sequence) or len(exts) != 2):
                raise TypeError("'{e}' entry must be a "
                    "sequence of length 2.".format(e=DimData.EXTENTS))

            self[DimData.EXTENTS] = [v for v in exts[0:2]]

        # Check that we've been given valid values
        self.validate()

    def is_expression(self):
        return (isinstance(self[DimData.GLOBAL_SIZE], str) or
            isinstance(self[DimData.LOCAL_SIZE], str) or
            isinstance(self[DimData.EXTENTS][0], str) or 
            isinstance(self[DimData.EXTENTS][1], str))

    def validate(self):
        """ Validate the contents of a dimension data dictionary """

        # Currently, we don't validate string expressions
        if self.is_expression():
            return

        ls, gs, E, name, zeros = (self[DimData.LOCAL_SIZE],
            self[DimData.GLOBAL_SIZE],
            self[DimData.EXTENTS],
            self[DimData.NAME],
            self[DimData.ZERO_VALID])

        # Sanity validate dimensions
        if ls > gs:
            raise ValueError("Dimension '{n}' "
                "local size {l} is greater than "
                "it's global size {g}".format(n=name, l=ls, g=gs))

        if E[1] - E[0] > ls: 
            raise ValueError("Dimension '{n}' "
                "extent range [{e0}, {e1}] ({r}) "
                "is greater than it's local size {l}. "
                "If this dimension is defined as "
                "an expression containing multiple "
                "dimensions, these extents may be "
                "much larger than the local size. "
                "Consider forcing the extents "
                "to [0,1] as meaningful extents "
                "are unlikely in these cases.".format(
                    n=name, l=ls, e0=E[0], e1=E[1], r=(E[1] - E[0])))

        if zeros:
            assert 0 <= E[0] <= E[1] <= gs, (
                "Dimension '{d}', global size {gs}, extents [{e0}, {e1}]"
                    .format(d=name, gs=gs, e0=E[0], e1=E[1]))
        else:
            assert 0 <= E[0] < E[1] <= gs, (
                "Dimension '{d}', global size {gs}, extents [{e0}, {e1}]"
                    .format(d=name, gs=gs, e0=E[0], e1=E[1]))    

