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

import collections
import numpy as np
import types

from weakref import WeakKeyDictionary
from attrdict import AttrDict

from hypercube.dims import create_dimension
from hypercube.expressions import (expand_expression_map,
    parse_expression as pe)
import hypercube.util as hcu

from tabulate import tabulate

class PropertyDescriptor(object):
    """ Descriptor class for properties """
    def __init__(self, record_key, default=None, ):
        self.default = default
        self.record_key = record_key
        self.data = WeakKeyDictionary()

    def __get__(self, instance, owner):
        return self.data.get(instance,self.default)

    def __set__(self, instance, value):
        dtype = instance._properties[self.record_key].dtype
        self.data[instance] = dtype(value)

    def __delete__(self, instance):
        del self.data[instance]

class HyperCube(object):
    """ Hypercube. """

    def __init__(self, *args, **kwargs):
        """
        Hypercube Constructor
        """

        # Dictionaries to store records about our
        # dimensions, arrays and properties
        self._dims = collections.OrderedDict()
        self._arrays = collections.OrderedDict()
        self._properties = collections.OrderedDict()

    def bytes_required(self):
        """ Returns the memory required by all arrays in bytes."""
        return np.sum([hcu.array_bytes(a) for a
            in self.arrays(reify=True).itervalues()])

    def mem_required(self):
        """ Return a string representation of the total memory required """
        return hcu.fmt_bytes(self.bytes_required())

    def register_dimension(self, name, dim_data, **kwargs):
        """
        Registers a dimension with this Solver object

        Arguments
        ---------
            dim_data : integer or Dimension
                if an integer, this will be used to
                define the global_size of the dimension
                and possibly other attributes if they are
                not present in kwargs.
                If a Dimension, it will be updated with
                any appropriate keyword arguments

        Keyword Arguments
        -----------------
            description : string
                The description for this dimension.
                e.g. 'Number of timesteps'.
            local_size : integer or None
                The local size of this dimension
                on this solver. If None, set to
                the global_size.
            lower_extent : integer or None
                The lower extent of this dimension
                within the global space
            upper_extent : integer or None
                The upper extent of this dimension
                within the global space
            zero_valid : boolean
                If True, this dimension may be zero-sized.

        Returns
        -------
        A Dimension object
        """

        if name in self._dims:
            raise AttributeError((
                "Attempted to register dimension '{n}'' "
                "as an attribute of the solver, but "
                "it already exists. Please choose "
                "a different name!").format(n=name))

        # Create the dimension dictionary
        D = self._dims[name] = create_dimension(name,
            dim_data, **kwargs)

        return D

    def register_dimensions(self, dim_list):
        """
        >>> slvr.register_dimensions([
            {'name' : 'ntime', 'local_size' : 10, 'extents' : [2, 7], 'safety': False },
            {'name' : 'na', 'local_size' : 3, 'extents' : [2, 7]},
            ])
        """

        for dim in dim_list:
            self.register_dimension(dim.name, dim)

    def update_dimensions(self, dim_list):
        """
        >>> slvr.update_dimensions([
            {'name' : 'ntime', 'local_size' : 10, 'extents' : [2, 7], 'safety': False },
            {'name' : 'na', 'local_size' : 3, 'extents' : [2, 7]},
            ])
        """
        for dim_data in dim_list:
            self.update_dimension(**dim_data)


    def update_dimension(self, name, **update_dict):
        """
        Update the dimension size and extents.

        Arguments
        ---------
            update_dict : dict
        """
        if not name:
            raise AttributeError("A dimension name is required to update "
                "a dimension. Update dictionary {u}."
                    .format(u=update_dict))

        dim = self._dims.get(name, None)

        # Sanity check dimension existence
        if not dim:
            montblanc.log.warn("'Dimension {n}' cannot be updated as it "
                "is not registered in the dimension dictionary."
                    .format(n=name))

            return

        dim.update(**update_dict)

    def _dim_attribute(self, attr, *args):
        """
        Returns a list of dimension attribute attr, for the
        dimensions specified as strings in args.

        ntime, nbl, nchan = slvr._dim_attribute('global_size', ntime, 'nbl', 'nchan')

        or

        ntime, nbl, nchan, nsrc = slvr._dim_attribute('global_size', 'ntime,nbl:nchan nsrc')
        """

        import re

        # If we got a single string argument, try splitting it by separators
        if len(args) == 1 and isinstance(args[0], str):
            args = (s.strip() for s in re.split(',|:|;| ', args[0]))

        # Create a variable template from this attribute
        # for use in parse_expressions below
        T = { d.name: getattr(d, attr) for d in self._dims.itervalues() }

        # Now get the specific attribute for each argument, parsing
        # any string expressions on the way
        result = [pe(getattr(self._dims[name], attr),
                variables=T, expand=True)
            for name in args]

        # Return single element if length one else entire list
        return result[0] if len(result) == 1 else result

    def dim_global_size_dict(self, reify=True):
        """ Returns a mapping of dimension name to global size """
        D = { d.name: d.global_size for d in self._dims.itervalues()}
        return expand_expression_map(D) if reify else D

    def dim_local_size_dict(self, reify=True):
        """ Returns a mapping of dimension name to local size """
        D = { d.name: d.local_size for d in self._dims.itervalues()}
        return expand_expression_map(D) if reify else D

    def dim_lower_extent_dict(self, reify=True):
        """ Returns a mapping of dimension name to lower_extent """
        D = { d.name: d.lower_extent for d in self._dims.itervalues()}
        return expand_expression_map(D) if reify else D

    def dim_upper_extent_dict(self, reify=True):
        """ Returns a mapping of dimension name to upper_extent """
        D = { d.name: d.upper_extent for d in self._dims.itervalues()}
        return expand_expression_map(D) if reify else D

    def dim_global_size(self, *args):
        """
        ntime, nbl, nchan = slvr.dim_global_size('ntime, 'nbl', 'nchan')

        or

        ntime, nbl, nchan, nsrc = slvr.dim_global_size('ntime,nbl:nchan nsrc')
        """

        return self._dim_attribute('global_size', *args)

    def dim_local_size(self, *args):
        """
        ntime, nbl, nchan = slvr.dim_local_size('ntime, 'nbl', 'nchan')

        or

        ntime, nbl, nchan, nsrc = slvr.dim_local_size('ntime,nbl:nchan nsrc')
        """

        return self._dim_attribute('local_size', *args)

    def dim_lower_extent(self, *args):
        """
        t_ex, bl_ex, ch_ex = slvr.dim_lower_extent('ntime, 'nbl', 'nchan')

        or

        t_ex, bl_ex, ch_ex, src_ex = slvr.dim_lower_extent('ntime,nbl:nchan nsrc')
        """

        return self._dim_attribute('lower_extent', *args)

    def dim_upper_extent(self, *args):
        """
        t_ex, bl_ex, ch_ex = slvr.dim_upper_extent('ntime, 'nbl', 'nchan')

        or

        t_ex, bl_ex, ch_ex, src_ex = slvr.dim_upper_extent('ntime,nbl:nchan nsrc')
        """

        return self._dim_attribute('upper_extent', *args)

    def dim_extents(self, *args):
        l = self.dim_lower_extent(*args)
        u = self.dim_upper_extent(*args)

        if isinstance(l, collections.Sequence):
            return zip(l, u)
        else:
            return (l, u)

    def register_array(self, name, shape, dtype, **kwargs):
        """
        Register an array with this Solver object.

        Arguments
        ----------
            name : string
                name of the array.
            shape : integer/string or tuple of integers/strings
                Shape of the array.
            dtype : data-type
                The data-type for the array.

        Returns
        -------
            A dictionary describing this array.
        """

        # Complain if array exists
        if name in self._arrays:
            raise ValueError(('Array %s is already registered '
                'on this solver object.') % name)

        # OK, create a record for this array
        A = self._arrays[name] = AttrDict(name=name,
            dtype=dtype, shape=shape,
            **kwargs)

        return A

    def register_arrays(self, array_list):
        """
        Register arrays using a list of dictionaries defining the arrays.

        The list should itself contain dictionaries. i.e.

        >>> D = [
            { 'name':'uvw', 'shape':(3,'ntime','nbl'),'dtype':np.float32 },
            { 'name':'lm', 'shape':(2,'nsrc'),'dtype':np.float32 }
        ]
        """
        for ary in array_list:
            self.register_array(**ary)

    def register_property(self, name, dtype, default, **kwargs):
        """
        Registers a property with this Solver object

        Arguments
        ----------
            name : string
                The name of the property.
            dtype : data-type
                The data-type of this property
            default :
                Default value for the property.

        Keyword Arguments
        -----------------
            setter : boolean or function
                if True, a default 'set_name' member is created, otherwise not.
                If a method, this is used instead.
            setter_docstring : string
                docstring for the default setter.
        """
        if name in self._properties:
            raise ValueError(('Property %s is already registered '
                'on this solver object.') % name)

        P = self._properties[name] = AttrDict(name=name,
            dtype=dtype, default=default)

        #if not hasattr(HyperCube, name):
        if not HyperCube.__dict__.has_key(name):
                # Create the descriptor for this property on the class instance
            setattr(HyperCube, name, PropertyDescriptor(record_key=name, default=default))

        # Set the descriptor on this object instance
        setattr(self, name, default)

        # Should we create a setter for this property?
        setter = kwargs.get('setter_method', True)
        setter_name = hcu.setter_name(name)

        # Yes, create a default setter
        if isinstance(setter, types.BooleanType) and setter is True:
            def set(self, value):
                setattr(self,name,value)

            setter_method = types.MethodType(set, self)
            setattr(self, setter_name, setter_method)

            # Set up the docstring, using the supplied one
            # if it is present, otherwise generating a default
            setter_docstring = kwargs.get('setter_docstring', None)
            getattr(setter_method, '__func__').__doc__ = \
                """ Sets property %s to value. """ % (name) \
                if setter_docstring is None else setter_docstring

        elif isinstance(setter, types.MethodType):
            setattr(self, setter_name, setter)
        else:
            raise TypeError, ('setter keyword argument set',
                ' to an invalid type %s' % (type(setter)))

        return P

    def register_properties(self, property_list):
        """
        Register properties using a list defining the properties.

        The dictionary should itself contain dictionaries. i.e.

        >>> D = [
            { 'name':'ref_wave','dtype':np.float32,
                'default':1.41e6, 'registrant':'solver' },
        ]
        """
        for prop in property_list:
            self.register_property(**prop)

    def properties(self):
        """ Returns a dictionary of properties """
        return self._properties

    def property(self, name):
        """ Returns a property """
        try:
            return self._properties[name]
        except KeyError:
            raise KeyError("Property '{n}' is not registered "
                "on this solver".format(n=name))

    def arrays(self, reify=False):
        """
        Returns a dictionary of arrays. If reify is True,
        it will replace any dimension within the array shape with
        the local_size of the dimension.
        """
        return (self._arrays if not reify else
            hcu.reify_arrays(self._arrays, self.dimensions(reify=True)))

    def array(self, name, reify=False):
        """
        Returns an array object. If reify is True,
        it will replace any dimension within the array shape with
        the local_size of the dimension.

        Reifying arrays "individually" is expensive since, in practice,
        all dimensions must be reified to handle dependent expressions.
        """

        # Complain if the array doesn't exist
        if name not in self._arrays:
            raise KeyError("Array '{n}' is not registered on this solver"
                .format(n=name))

        # Just return the array if we're not reifying
        if not reify:
            return self._arrays[name]

        return hcu.reify_arrays({name : self._arrays[name]},
            self.dimensions(reify=True))[name]

    def dimensions(self, reify=False, copy=True):
        """
        Return a dictionary of dimensions

        Keyword Arguments
        -----------------
            reify : boolean
                if True, converts any expressions in the dimension
                information to integers.
        """

        return hcu.reify_dims(self._dims) if reify else self._dims

    def dimension(self, name, reify=False):
        """
        Returns a dimension object.

        Reifying dimensions "individually" is expensive since, in practice,
        all dimensions must be reified to handle dependent expressions.
        """
        # Complain if the array doesn't exist
        if name not in self._dims:
            raise KeyError("Dimension '{n}' is not registered on this solver"
                .format(n=name))

        # Just return the array if we're not reifying
        if not reify:
            return self._dims[name]

        # Reifies everything just to get this dimension, expensive
        return hcu.reify_dims(self._dims)[name]

    def gen_dimension_table(self):
        """ 2D array describing each registered dimension together with headers - for use in __str__ """
        headers = ['Dimension Name', 'Description', 'Global Size', 'Local Size', 'Extents']

        table = []
        for dimval in sorted(self.dimensions(reify=True).itervalues(),
                             key=lambda dval: dval.name.upper()):
            table.append([dimval.name, dimval.description, dimval.global_size, dimval.local_size, (dimval.lower_extent, dimval.upper_extent)])
        return table, headers

    def gen_array_table(self):
        """ 2D array describing each registered array together with headers - for use in __str__ """
        headers = ['Array Name', 'Size', 'Type', 'Shape']

        # Reify arrays to work out their actual size
        reified_arrays = self.arrays(reify=True)

        table = []
        for arrval in sorted(self.arrays().itervalues(),
                             key=lambda aval: aval.name.upper()):
            # Get the actual size of the array
            nbytes = hcu.array_bytes(reified_arrays[arrval.name])
            # Print shape tuples without spaces and single quotes
            sshape = '(%s)' % (','.join(map(str, arrval.shape)),)
            table.append([arrval.name, hcu.fmt_bytes(nbytes), np.dtype(arrval.dtype).name, sshape])
        return table, headers

    def gen_property_table(self):
        """ 2D array describing each registered property together with headers - for use in __str__ """
        headers = ['Property Name', 'Type', 'Value', 'Default Value']

        table = []
        for propval in sorted(self._properties.itervalues(),
                              key=lambda pval: pval.name.upper()):
            table.append([propval.name, np.dtype(propval.dtype).name, getattr(self, propval.name), propval.default])
        return table, headers

    def __str__(self):
        """ Outputs a string representation of this object """

        result = ''

        if len(self._dims) > 0:
            table, headers = self.gen_dimension_table()
            assert headers == ['Dimension Name', 'Description', 'Global Size', 'Local Size', 'Extents']
            result += "Registered Dimensions:\n%s\n\n" % (tabulate(table, headers=headers),)

        if len(self._arrays) > 0:
            table, headers = self.gen_array_table()
            table.append(['Local Memory Usage', self.mem_required(), '', ''])
            assert headers == ['Array Name', 'Size', 'Type', 'Shape']
            result += "Registered Arrays:\n%s\n\n" % (tabulate(table, headers=headers),)

        if len(self._properties) > 0:
            table, headers = self.gen_property_table()
            assert headers == ['Property Name', 'Type', 'Value', 'Default Value']
            result += "Registered Properties:\n%s\n\n" % (tabulate(table, headers=headers),)

        return result
