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

DEFAULT_DESCRIPTION = 'The FOURTH dimension!'

def create_dimension(name, dim_data, **kwargs):
    if isinstance(dim_data, Dimension):
        dim = dim_data.copy()
        dim.update(**kwargs)
    else:
        dim = Dimension(name, dim_data, **kwargs)

    return dim

class Dimension(object):
    __slots__ = ['_name', '_global_size',
        '_lower_extent', '_upper_extent', '_description']

    def __init__(self, name, global_size,
            lower_extent=None, upper_extent=None,
            description=None):
        """
        Create a Dimension from supplied arguments
        """
        self._name = name
        self._global_size = global_size
        self._lower_extent = 0 if lower_extent is None else lower_extent
        self._upper_extent = (global_size if upper_extent is None
                                    else upper_extent)
        self._description = (DEFAULT_DESCRIPTION if description is None
                                    else description)

    def copy(self):
        return Dimension(self._name, self._global_size,
            lower_extent=self._lower_extent,
            upper_extent=self._upper_extent,
            description=self._description)

    @property
    def name(self):
        return self._name

    @property
    def global_size(self):
        return self._global_size

    @property
    def lower_extent(self):
        return self._lower_extent

    @property
    def upper_extent(self):
        return self._upper_extent

    @property
    def extent_size(self):
        return self.upper_extent - self.lower_extent

    @property
    def description(self):
        return self._description

    def __eq__(self, other):
        # Note description is left out
        return (self.name == other.name and
            self.global_size == other.global_size and
            self.lower_extent == other.lower_extent and
            self.upper_extent == other.upper_extent)

    def __ne__(self, other):
        return not self.__eq__(other)

    def update(self, global_size=None, lower_extent=None, upper_extent=None,
        description=None):

        if global_size is not None: self._global_size = global_size
        if lower_extent is not None: self._lower_extent = lower_extent
        if upper_extent is not None: self._upper_extent = upper_extent
        if description is not None: self._description = description

        # Check that we've been given valid values
        self.validate()

    def validate(self):
        """ Validate the contents of a dimension data dictionary """

        extents_valid = (0 <= self.lower_extent <= self.upper_extent
            <= self.global_size)

        if not extents_valid:
            raise ValueError("Dimension '{d}' fails 0 <= {el} <= {eu} <= {gs}"
                .format(d=self.name, gs=self.global_size,
                    el=self.lower_extent, eu=self.upper_extent))

    def __str__(self):
        return ("['{n}': global: {gs} lower: {el} upper: {eu}]").format(
                n=self.name, gs=self.global_size,
                el=self.lower_extent, eu=self.upper_extent)

