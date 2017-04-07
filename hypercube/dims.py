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
    """

    Parameters
    ----------
    name : str
        Dimension name

    dim_data : int or :class:`~hypercube.dims.Dimension`
        if an integer, this will be used to
        define the global_size of the dimension
        and possibly other attributes if they are
        not present in kwargs.
        If a Dimension, it will be updated with
        any appropriate keyword arguments
    
    Keywords
    --------
    description : str
        The description for this dimension.
        e.g. 'Number of timesteps'.
    lower_extent : int
        The lower extent of this dimension
        within the global space
    upper_extent : int
        The upper extent of this dimension
        within the global space
        

    Returns
    -------
    :class:`~hypercube.dims.Dimension`
        A hypercube :class:`~hypercube.dims.Dimension`


    """
    if isinstance(dim_data, Dimension):
        dim = dim_data.copy()
        dim.update(**kwargs)
    else:
        dim = Dimension(name, dim_data, **kwargs)

    return dim

class Dimension(object):
    """
    The Dimension class describes a hypercube dimension.
    """
    __slots__ = ['_name', '_global_size',
        '_lower_extent', '_upper_extent', '_description']

    def __init__(self, name, global_size,
            lower_extent=None, upper_extent=None,
            description=None):
        """
        Create a Dimension from supplied arguments

        :param name: Dimension name
        :type name: str
        :param global_size: Global dimension size
        :type global_size: int
        :param local_extent: Lower dimension extent
        :type local_extent: int
        :param upper_extent: Upper dimension extent
        :type upper_extent: int
        :param description: Dimension description
        :type description: str

        """
        self._name = name
        self._global_size = global_size
        self._lower_extent = 0 if lower_extent is None else lower_extent
        self._upper_extent = (global_size if upper_extent is None
                                    else upper_extent)
        self._description = (DEFAULT_DESCRIPTION if description is None
                                    else description)

    def copy(self):
        """
        Returns
        -------
            A copy of the dimension
        """
        return Dimension(self._name, self._global_size,
            lower_extent=self._lower_extent,
            upper_extent=self._upper_extent,
            description=self._description)

    @property
    def name(self):
        """ Dimension name """
        return self._name

    @property
    def global_size(self):
        """ Global dimension size """
        return self._global_size

    @property
    def lower_extent(self):
        """Lower dimension extent"""
        return self._lower_extent

    @property
    def upper_extent(self):
        """ Upper dimension extent """
        return self._upper_extent

    @property
    def extent_size(self):
        """
        Size of the dimension extents.
        Equal to :obj:`~Dimension.upper_extent` - :obj:`~Dimension.lower_extent`
        """
        return self.upper_extent - self.lower_extent

    @property
    def description(self):
        """ Dimension description """
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
        """
        Update the dimension properties

        Parameters
        ----------
        global_size : int
            Global dimension size (Default value = None)
        lower_extent : int
            Lower dimension extent (Default value = None)
        upper_extent : int
            Upper dimension extent (Default value = None)
        description : str
            Dimension description (Default value = None)
        """

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

